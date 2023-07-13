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

from typing import Dict, Optional, Tuple
from abc import ABC

import random

from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck
from nemo.core.classes import NeuralModule

from nemo.utils import logging


class TextPreprocessor(NeuralModule, ABC):
    """Simple Embedding module to convert codes to latent vectors
    Args:
        embedding_dim: dimension of the embedding
        embedding_out_dim: dimension of the output embedding
        padding_idx: padding index
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int=512,
        embedding_out_dim: int=512,
        padding_idx: Optional[int]=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        if padding_idx is not None:
            self.embedding = torch.nn.Embedding(
                num_embeddings=num_embeddings+1,      # +1 for padding_idx
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = torch.nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
            )
        proj = torch.nn.Linear(embedding_dim, embedding_out_dim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(inplace=False), proj)


    def forward(self, 
                input_signal: torch.Tensor,
                length: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a batch of input codes to a batch of embeddings
        Args:
            input_signal: input codes of shape (B, D, T)
            length: length of each input code sequence of shape (B,)
        Returns:
            output: embeddings of shape (B, embedding_dim, T)
            output_length: length of each output embedding sequence of shape (B,)
        """
        # Apply padding before embedding.
        embed = self.embedding(input_signal)
        output = self.proj(embed)
        output = torch.transpose(output, -1, -2)    # to be consistent with other preprocessors
        # [B, D, T]
        return output, length



class SpeechEncDecEnCodecTextSelfSupervisedModel(SpeechEncDecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.codebook_size = self._cfg.model_defaults.codebook_size
        self.n_codebooks = self._cfg.model_defaults.n_codebooks_to_use

        self.decoder_ssl = None
        self.heads = nn.ModuleList([self.from_config_dict(self._cfg.decoder) for _ in range(self.n_codebooks)])  # Plus one for the text head
        print(self._train_dl.dataset.manifest_processor.parser._labels)
        self.text_preprocessor = TextPreprocessor(num_embeddings=len(self._cfg.labels),embedding_dim=self._cfg.preprocessor.embedding_out_dim, embedding_out_dim=self.encoder.d_model, padding_idx=self._train_dl.dataset.manifest_processor.pad_id)
        self.text_head = self.from_config_dict(self._cfg.text_decoder)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        # Instantiate tarred dataset loader or normal dataset loader
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
        dataset = audio_to_text_dataset.get_audioCodes_to_text_char_dataset(config=config, augmentor=augmentor)
        collate_fn = dataset.collate_fn
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def forward_text(self, input_string=None, input_string_length=None):
        torch.autograd.set_detect_anomaly(True)
        processed_string, processed_string_len = self.text_preprocessor(input_string, input_string_length)
        # We use the stacked codebooks as spectrogram targets.
        # For smaller inputs
        if self.dropout_features:
            processed_string = self.dropout_features(processed_string)
        if self.apply_masking:
            processed_string = self.spec_augmentation(input_spec=processed_string, length=processed_string_len)
        tokens = input_string.detach().clone()
        masked_tokens = processed_string.detach()
        token_masks = torch.logical_and(masked_tokens < 1e-5, masked_tokens > -1e-5).float()
        for idx, proc_len in enumerate(processed_string_len):
            token_masks[idx, :, proc_len:] = 0.0
        encoded, encoded_len = self.encoder(audio_signal=processed_string, length=processed_string_len, skip_pre_encode=True)
        return tokens, token_masks, encoded, encoded_len


    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
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
        # Reset access registry
        if self.is_access_enabled():
            self.reset_registry()
        # Check for special flag for validation step
        if hasattr(self, '_in_validation_step'):
            in_validation_step = self._in_validation_step
        else:
            in_validation_step = False

        # reset module registry from AccessMixin
        if (
            (self.training or in_validation_step)
            and self.decoder_losses is not None
            and self.output_from_layer is not None
            and len(self.output_from_layer) > 0
        ):
            layer_names = list(self.output_from_layer.values())
            register_layer = any([name is not None for name in layer_names])

            if register_layer:
                self.access_cfg['save_encoder_tensors'] = True
                self.set_access_enabled(access_enabled=True)

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
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)
        if self.apply_masking:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
        masked_spectrograms = processed_signal.detach()
        spec_masks = torch.logical_and(masked_spectrograms < 1e-5, masked_spectrograms > -1e-5).float()
        for idx, proc_len in enumerate(processed_signal_length):
            spec_masks[idx, :, proc_len:] = 0.0
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return spectrograms, spec_masks, encoded, encoded_len

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None):
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
        loss_val_dict = {}
        loss_value = encoded.new_zeros(1)
        # -> BxTxD
        for idx in range(self.n_codebooks):
            logits = self.heads[idx](encoder_output=encoded)
            curr_loss = self.loss(
                spec_masks=spec_masks,
                decoder_outputs=nn.functional.log_softmax(logits, -1),
                targets=spectrograms[:,idx,:] - self.codebook_size*idx, # since encodings are scaled
                decoder_lengths=None,
                target_lengths=None,
            )
            loss_val_dict[f"head_{idx}"] = curr_loss
            loss_value = loss_value + curr_loss
        return loss_value/self.n_codebooks, loss_val_dict
    
    def text_decoder_loss_step(self, tokens, token_masks, encoded, encoded_len, targets=None, target_lengths=None):
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
        loss_val_dict = {}
        loss_value = encoded.new_zeros(1)
        # -> BxTxD
        logits = self.text_head(encoder_output=encoded)
        curr_loss = self.loss(
            spec_masks=token_masks,
            decoder_outputs=nn.functional.log_softmax(logits, -1),
            targets=tokens, # since encodings are scaled
        )
        loss_val_dict[f"text_head"] = curr_loss
        loss_value = loss_value + curr_loss
        return loss_value, loss_val_dict

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'D', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "targets": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "target_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):            
        # Set flag to register tensors
        self._in_validation_step = True

        signal, signal_len, targets, target_lengths = batch
        spectrograms, spec_masks, encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )
        tokens, token_masks, encoded_text, encoded_text_len = self.forward_text(
            input_string=targets, input_string_length=target_lengths
        )

        loss_value, loss_val_dict = self.decoder_loss_step(
            spectrograms, spec_masks, encoded, encoded_len,
        )
        text_loss_value, text_loss_val_dict = self.text_decoder_loss_step(
            tokens, token_masks, encoded_text, encoded_text_len
        )

        loss_value = loss_value + text_loss_value
        loss_val_dict.update(text_loss_val_dict)

        if self.feat_pen:
            loss_value += self.feat_pen

        # reset access registry
        self.reset_registry()
        del self._in_validation_step
        loss_val_dict["val_loss"] = loss_value
        return loss_val_dict

    # PTL-specific methods
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        loss_dict = {}
        for key in outputs[0].keys():
            loss_dict[key] = torch.stack([x[key] for x in outputs]).mean()
        loss_dict['log'] = {k: v for k, v in loss_dict.items()}
        return loss_dict

    def training_step(self, batch, batch_nb):
        signal, signal_len, targets, target_lengths = batch

        spectrograms, spec_masks, encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len,
        )
        audio_loss_value, loss_val_dict = self.decoder_loss_step(
            spectrograms, spec_masks, encoded, encoded_len,
        )
        
        tokens, token_masks, encoded_text, encoded_text_len = self.forward_text(
            input_string=targets, input_string_length=target_lengths
        )
        text_loss_value, text_loss_val_dict = self.text_decoder_loss_step(
            tokens, token_masks, encoded_text, encoded_text_len
        )
        loss_val_dict.update(text_loss_val_dict)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
        }

        for loss_name, loss_val in loss_val_dict.items():
            tensorboard_logs['train_' + loss_name] = loss_val
        return {'loss': audio_loss_value + text_loss_value, 'log': tensorboard_logs}
