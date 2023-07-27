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

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.core import typecheck

#typecheck.set_typecheck_enabled(enabled=False) 

from nemo.collections.asr.models.ssl_encodec_models import SpeechEncDecEnCodecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck


class SpeechEncDecEnCodecNewMaskSelfSupervisedModel(SpeechEncDecEnCodecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""
    
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.decoder_weights = self._cfg.model_defaults.get("decoder_weights", [1 for _ in range(self.n_codebooks)])
        assert self.preprocessor.pad_value == self.spec_augmentation.mask_value

    def _quantize(self, x, q):
        embed = q.t()
        dist = torch.stack([-(
            x[idx].pow(2).sum(1, keepdim=True)
            - 2 * x[idx] @ embed 
            + embed.pow(2).sum(0, keepdim=True)
        ) for idx in range(x.shape[0])], dim=0)
        return dist


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
        # We use the stacked codebooks as spectrogram targets.
        spectrograms = input_signal.detach().clone()
        if self.apply_masking:
            input_signal = self.spec_augmentation(input_spec=input_signal, length=input_signal_length)
        masked_spectrograms = input_signal.detach().clone()
        spec_masks = (masked_spectrograms == self.spec_augmentation.mask_value).float()
        for idx, proc_len in enumerate(input_signal_length):
            spec_masks[idx, :, proc_len:] = 0.0
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)
        spec_masks = torch.nn.functional.pad(spec_masks, (0, processed_signal.shape[-1] - spec_masks.shape[-1]), value=0.0)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return spectrograms, spec_masks, encoded, encoded_len

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None):
        loss_value, loss_val_dict = encoded.new_zeros(self.n_codebooks), {}
        # -> BxTxD
        for idx in range(self.n_codebooks):
            if torch.any(spec_masks[:,idx,:]):
                decoded = self.heads[idx](encoder_output=encoded)
                if self.decoding_mode == "quantize":
                    codebook_q = self.decoder_ssl[idx*self.codebook_size:(idx+1)*self.codebook_size]
                    logits = self._quantize(decoded, codebook_q)
                else:
                    logits = self.decoder_ssl(decoded)
                curr_loss = self.loss(
                    spec_masks=spec_masks[:,idx,:].unsqueeze(dim=1),
                    decoder_outputs=nn.functional.log_softmax(logits, -1),
                    targets= spectrograms[:,idx,:] if self.decoding_mode == "embedding" else spectrograms[:,idx,:] - self.codebook_size*idx , # base decoding only projects to codebook size while 'embedding' projects over all codebook values
                )
                loss_value[idx] += curr_loss*self.decoder_weights[idx]
        return loss_value[loss_value != 0].mean(), loss_val_dict  # nonzero loss values only

