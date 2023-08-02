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

from nemo.core.classes.module import NeuralModule

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck

from nemo.utils import logging


class Quantizer(NeuralModule):
    def __init__(self, embed, q, n):
        super().__init__()
        self.weight = embed
        self.q = q
        self.n = n

    def forward(self, x):
        embed = self.weight[self.q*self.n:(self.q+1)*self.n].t()
        dist = torch.stack([-(
            x[idx].pow(2).sum(1, keepdim=True)
            - 2 * x[idx] @ embed
            + embed.pow(2).sum(0, keepdim=True)
        ) for idx in range(x.shape[0])], dim=0)
        return dist
    
class EmbProj(NeuralModule):
    def __init__(self, embed, q, n):
        super().__init__()
        self.weight = embed
        self.q = q
        self.n = n
        self.bias = nn.parameter.Parameter(torch.zeros(n))
        nn.init.uniform_(self.bias)

    def forward(self, x):
        embed = self.weight[self.q*self.n:(self.q+1)*self.n]
        return torch.matmul(x, embed.t()) + self.bias

class SpeechEncDecEnCodecSelfSupervisedModel(SpeechEncDecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.codebook_size = cfg.model_defaults.codebook_size
        self.n_codebooks = cfg.model_defaults.n_codebooks_to_use
        self.output_dim = cfg.decoder.feat_out

        cfg.decoder.feat_out *= self.n_codebooks  # predicting n codebooks at a time. 
        super().__init__(cfg=cfg, trainer=trainer)

        if self._cfg.preprocessor.get("init_from_encodec", False):
            self._cfg.preprocessor.init_from_encodec = None  # To stop from reloading every training.

        codebook_weights = self._cfg.model_defaults.get("codebook_weights", [1 for _ in range(self.n_codebooks)])
        self.register_buffer("codebook_weights", torch.tensor(codebook_weights, requires_grad=False, dtype=torch.float32))

        self.decode_mode = self._cfg.model_defaults.get("decode_mode", "base")
        if self.decode_mode == "base":
            self.heads = nn.ModuleList(nn.Linear(self.output_dim, self.codebook_size) for _ in range(self.n_codebooks)) # Just a dummy value so decoder looks nice
        else:
            codebooks = self.preprocessor.embedding.weight #[self.preprocessor.embedding.weight[idx*self.codebook_size:(idx+1)*self.codebook_size] for idx in range(self.n_codebooks)]
            if self.decode_mode == "quantize":
                self.heads = nn.ModuleList(Quantizer(codebooks, q, self.codebook_size) for q in range(self.n_codebooks))
            elif self.decode_mode == "embed":
                self.heads = nn.ModuleList(EmbProj(codebooks, q, self.codebook_size) for q in range(self.n_codebooks))
            else:
                assert False
        
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

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            logging.warning(f"Dali dataset is not supported")
            return None
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
            dataset = audio_to_text_dataset.get_tarred_audioCodes_dataset(
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
        
            dataset = audio_to_text_dataset.get_audioCodes_to_text_char_dataset(config=config, augmentor=augmentor)

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
        with torch.no_grad():
            input_signal = self.spec_augmentation(input_spec=input_signal, length=input_signal_length)
            spec_masks = (input_signal != spectrograms).float()
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        if self.dropout_features:
            processed_signal = self.dropout_features(processed_signal)
        # Processed signal is padded to allow subsampling
        spec_masks = torch.nn.functional.pad(spec_masks, (0, processed_signal.shape[-1] - spec_masks.shape[-1]), value=0.0)
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return spectrograms, spec_masks, encoded, encoded_len

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None):
        loss_val_dict = {}
        loss_value = encoded.new_zeros(self.n_codebooks)
        
        decoded = self.decoder_ssl(encoder_output=encoded)
        decoded_q = decoded.view(decoded.shape[0], decoded.shape[1], self.n_codebooks, self.output_dim)
        # -> BxTxD
        for idx in range(self.n_codebooks):
            if torch.any(spec_masks[:,idx,:]):
                logits = self.heads[idx](decoded_q[:,:,idx,:])
                curr_loss = self.loss(
                    spec_masks=spec_masks[:,idx,:].unsqueeze(dim=1),
                    decoder_outputs=nn.functional.log_softmax(logits, -1),
                    targets= spectrograms[:,idx,:] - self.codebook_size*idx,  # We only calculate loss in relation to codebook
                )
                loss_val_dict[f"head_{idx}"] = curr_loss
                loss_value[idx] = curr_loss
        return torch.dot(loss_value, self.codebook_weights), loss_val_dict  # nonzero loss values only
