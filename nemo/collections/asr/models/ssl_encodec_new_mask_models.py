# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import random

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.core import typecheck

#typecheck.set_typecheck_enabled(enabled=False) 

from nemo.core.neural_types import (
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    AcousticEncodedRepresentation,
    SpectrogramType,
)

from nemo.collections.asr.models.ssl_encodec_models import SpeechEncDecEnCodecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck

from nemo.utils import logging


class SpeechEncDecEnCodecNewMaskSelfSupervisedModel(SpeechEncDecEnCodecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""
    
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "spectrograms": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "spec_masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "encoded": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
            "selected_heads": NeuralType(tuple('C'), LengthsType())
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 4 elements -
            1) Processed spectrograms of shape [B, D, T].
            2) Masks applied to spectrograms of shape [B, D, T].
            3) The encoded features tensor of shape [B, D, T].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        # Check for special flag for validation step
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )
        # We use the stacked codebooks as spectrogram targets.
        spectrograms = input_signal.detach().clone()
        # For smaller inputs
        if self.preprocessor.n_codebooks_to_use < self.n_codebooks:
            n = self.preprocessor.n_codebooks_to_use
            input_signal = input_signal[:,:n,:]
            input_signal = torch.where(input_signal == self.n_codebooks*self.codebook_size, self.preprocessor.pad_value, input_signal)
        
        if self.apply_masking:
            codes = self.target_codes + random.sample(self.valid_codes, self.n_decoders)
            input_signal = self.spec_augmentation(input_spec=input_signal, length=input_signal_length, codes=codes)
        masked_spectrograms = input_signal.detach()
        spec_masks = (masked_spectrograms == self.spec_augmentation.padding_idx).float()
        for idx, proc_len in enumerate(input_signal_length):
            spec_masks[idx, :, proc_len:] = 0.0

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)

        spec_masks = torch.nn.functional.pad(spec_masks, (0, processed_signal.shape[-1] - spec_masks.shape[-1]), value=0.0)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return spectrograms, spec_masks, encoded, encoded_len, codes

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None, selected_heads=None):
        """
        Forward pass through all decoders and calculate corresponding losses.
        Args:
            spectrograms: Processed spectrograms of shape [B, D, T].
            spec_masks: Masks applied to spectrograms of shape [B, D, T].
            encoded: The encoded features tensor of shape [B, D, T].
            encoded_len: The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            targets: Optional target labels of shape [B, T]
            target_lengths: Optional target label lengths of shape [B]

        Returns:
            A tuple of 2 elements -
            1) Total sum of losses weighted by corresponding loss_alphas
            2) Dictionary of unweighted losses
        """
        #loss_val_dict = {}
        loss_value = encoded.new_zeros(1)
        # -> BxTxD
        denom = len(selected_heads)
        for idx in selected_heads:
            logits = self.heads[idx](encoder_output=encoded)
            curr_loss = self.loss(
                spec_masks=spec_masks[:,idx,:].unsqueeze(dim=1),
                decoder_outputs=nn.functional.log_softmax(logits, -1),
                targets=spectrograms[:,idx,:] - self.codebook_size*idx, # since encodings are scaled
                decoder_lengths=None,
                target_lengths=None,
            )
            #loss_val_dict[f"head_{idx}"] = curr_loss
            loss_value = loss_value + curr_loss
        return loss_value/denom #loss_val_dict

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, targets, target_lengths = batch
        spectrograms, spec_masks, encoded, encoded_len, target_codes = self.forward(
            input_signal=signal, input_signal_length=signal_len,
        )

        if hasattr(self.loss, "set_num_updates"):
            self.loss.set_num_updates(self.trainer.global_step)

        loss_value = self.decoder_loss_step(
            spectrograms=spectrograms, spec_masks=spec_masks, encoded=encoded, encoded_len=encoded_len, targets=targets, target_lengths=target_lengths, selected_heads=target_codes
        )

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
        }

        # for loss_name, loss_val in loss_val_dict.items():
        #     tensorboard_logs['train_' + loss_name] = loss_val

        if self.feat_pen:
            loss_value += self.feat_pen
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):            
        # Set flag to register tensors
        self._in_validation_step = True
        signal, signal_len, targets, target_lengths = batch
        spectrograms, spec_masks, encoded, encoded_len, _ = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )
        loss_value = self.decoder_loss_step(spectrograms=spectrograms, spec_masks=spec_masks, encoded=encoded, encoded_len=encoded_len, targets=targets, target_lengths=target_lengths, selected_heads=self.target_codes + self.valid_codes)
        if self.feat_pen:
            loss_value += self.feat_pen
        del self._in_validation_step
        return {
            'val_loss': loss_value,
        }
