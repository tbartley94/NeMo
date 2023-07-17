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

typecheck.set_typecheck_enabled(enabled=False) 

from nemo.core.neural_types import (
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck

from nemo.utils import logging


class SpeechEncDecEnCodecSelfSupervisedModel(SpeechEncDecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.codebook_size = self._cfg.model_defaults.codebook_size
        self.n_codebooks = self._cfg.model_defaults.n_codebooks_to_use
        # Checks which heads to include and exclude. If passed, n_decoders is treated as 'additional' decoders.
        exclude_codes = self._cfg.model_defaults.get("exclude_codes", [])
        target_codes =  self._cfg.model_defaults.get("target_codes", [])
        valid_codes = [idx for idx in range(self.n_codebooks) if idx not in target_codes and idx not in exclude_codes]

        self.valid_codes = valid_codes
        self.target_codes = target_codes
        self.n_decoders = self._cfg.model_defaults.get("n_decoders_to_use", len(valid_codes))

        self._cfg.decoder.feat_out = self.codebook_size
        self.heads = nn.ModuleList([self.from_config_dict(self._cfg.decoder) for _ in range(self.n_codebooks)])
        self.decoder_ssl = None

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None
            if config.get('audio_type', 'not_codes') == 'codes':
                dataset = audio_to_text_dataset.get_audioCodes_to_text_char_dataset(config=config, augmentor=augmentor)
            else:
                dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

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
        loss_val_dict = {}
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
            loss_val_dict[f"head_{idx}"] = curr_loss
            loss_value = loss_value + curr_loss
        return loss_value/denom, loss_val_dict

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, targets, target_lengths = batch
        spectrograms, spec_masks, encoded, encoded_len, target_codes = self.forward(
            input_signal=signal, input_signal_length=signal_len,
        )

        if hasattr(self.loss, "set_num_updates"):
            self.loss.set_num_updates(self.trainer.global_step)

        loss_value, loss_val_dict = self.decoder_loss_step(
            spectrograms=spectrograms, spec_masks=spec_masks, encoded=encoded, encoded_len=encoded_len, targets=targets, target_lengths=target_lengths, selected_heads=target_codes
        )

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
        }

        for loss_name, loss_val in loss_val_dict.items():
            tensorboard_logs['train_' + loss_name] = loss_val

        if self.feat_pen:
            loss_value += self.feat_pen

        # Reset access registry
        self.reset_registry()

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):            
        # Set flag to register tensors
        self._in_validation_step = True

        signal, signal_len, targets, target_lengths = batch
        spectrograms, spec_masks, encoded, encoded_len, target_codes = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )

        loss_value, loss_val_dict = self.decoder_loss_step(spectrograms=spectrograms, spec_masks=spec_masks, encoded=encoded, encoded_len=encoded_len, targets=targets, target_lengths=target_lengths, selected_heads=self.target_codes + self.valid_codes)

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
