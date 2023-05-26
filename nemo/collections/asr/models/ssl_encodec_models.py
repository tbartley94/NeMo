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

from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin, set_access_cfg
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    MaskType,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging


class SpeechEncDecEnCodecSelfSupervisedModel(SpeechEncDecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)


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
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "encoded": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_len": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of audio codebook entries,
                of shape [B, T*N]. T  represents number of frames, with N codebooks for every
                1 frame of compressed audio.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) Masks applied to encodings of shape [B, D, T].
            2) The encoded features tensor of shape [B, D, T].
            3) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
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

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        if self.pen_factor:
            self.feat_pen = processed_signal.float().pow(2).mean() * self.pen_factor

        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)

        if self.apply_masking:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        masked_signal = processed_signal.detach().clone()
        signal_masks = torch.logical_and(masked_signal < 1e-5, masked_signal > -1e-5).float()
        for idx, proc_len in enumerate(processed_signal_length):
            signal_masks[idx, :, proc_len:] = 0.0

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return signal_masks, encoded, encoded_len

    def decoder_loss_step(self, masks, encoded, encoded_len, targets=None, target_lengths=None):
        """
        Forward pass through all decoders and calculate corresponding losses.
        Args:
            masks: Masks applied to encoding of shape [B, D, T].
            encoded: The encoded features tensor of shape [B, D, T].
            encoded_len: The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            targets: Optional target labels of shape [B, T]
            target_lengths: Optional target label lengths of shape [B]

        Returns:
            A tuple of 2 elements -
            1) Total sum of losses weighted by corresponding loss_alphas
            2) Dictionary of unweighted losses
        """


        # Reshaping to get contiguous codes across time blocks.
        b, t = targets.shape
        masks_flat = masks.transpose(-1,-2).reshape(b, t, -1).transpose(-1,-2)
        encoded_flat = encoded.transpose(-1,-2).reshape(b, t, -1).transpose(-1,-2)


        # IGNORE: For verification
        # for idx in range(b):
        #     for time_step in range(int(t/N)):
        #         old = masks[idx, :, time_step]
        #         new = torch.cat([masks_flat[idx, :, time_step*N+i] for i in range(N)])
        #         assert torch.equal(new, old)



        outputs = self.decoder_ssl(encoder_output=encoded_flat)
        loss_value = self.loss(
            spec_masks=masks_flat,
            decoder_outputs=outputs,
            targets=targets,
            decoder_lengths=None, # not needed
            target_lengths=None, # not needed
        )

        return loss_value, {}

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, _, _ = batch
        masks, encoded, encoded_len = self.forward(
            input_signal=signal, input_signal_length=signal_len,
        )
        if hasattr(self.loss, "set_num_updates"):
            self.loss.set_num_updates(self.trainer.global_step)

        loss_value, loss_val_dict = self.decoder_loss_step(
            masks, encoded, encoded_len, signal, signal_len
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

        signal, signal_len, _, _ = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            masks, encoded, encoded_len = self.forward(
                processed_signal=signal, processed_signal_length=signal_len,
            )
        else:
            masks, encoded, encoded_len = self.forward(
                input_signal=signal, input_signal_length=signal_len,
            )

        loss_value, _ = self.decoder_loss_step(masks, encoded, encoded_len, signal, signal_len)

        if self.feat_pen:
            loss_value += self.feat_pen

        # reset access registry
        self.reset_registry()
        del self._in_validation_step

        return {
            'val_loss': loss_value,
        }

