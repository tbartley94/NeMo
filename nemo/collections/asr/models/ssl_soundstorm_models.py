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

from nemo.collections.asr.models.ssl_encodec_models import SpeechEncDecEnCodecSelfSupervisedModel
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import typecheck

from nemo.utils import logging


class SpeechEncDecSoundStormSelfSupervisedModel(SpeechEncDecEnCodecSelfSupervisedModel):
    """Base class for encoder-decoder models used for self-supervised encoder pre-training"""

    def decoder_loss_step(self, spectrograms, spec_masks, encoded, encoded_len, targets=None, target_lengths=None):
        loss_val_dict = {}
        loss_value = encoded.new_zeros(self.n_codebooks)
        
        decoded = self.decoder_ssl(encoder_output=encoded)
        # -> BxTxD*N
        decoded_q = decoded.view(decoded.shape[0], decoded.shape[1], self.n_codebooks, self.output_dim)
        # -> BxTxNxD
        for idx in range(self.n_codebooks):
            if torch.any(spec_masks[:,idx,:]):
                logits = self.heads[idx](decoded_q[:,:,idx,:])
                curr_loss = self.loss(
                    spec_masks=spec_masks[:,idx,:].unsqueeze(dim=1),  # For compatibility with loss
                    decoder_outputs=nn.functional.log_softmax(logits, -1),
                    targets= spectrograms[:,idx,:] - self.codebook_size*idx,  # We only calculate loss in relation to codebook
                )
                loss_val_dict[f"head_{idx}"] = curr_loss
                loss_value[idx] = curr_loss
                break # only need first code, rest are masked
        return torch.dot(loss_value, self.codebook_weights), loss_val_dict  # nonzero loss values only
