# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data.text_to_text_lhotse_prompted import (
    PromptedTextToTextLhotseDataset,
    PromptedTextToTextMiniBatch,
)
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.parts.submodules.multitask_decoding import lens_to_mask
from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, LogprobsType, MaskType, NeuralType

__all__ = ['DecMultiTaskModel']


class DecMultiTaskModel(EncDecMultiTaskModel):

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_signal": NeuralType(('B', 'T'), LabelsType, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "transcript": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "transcript_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "prompt": NeuralType(('B', 'T'), LabelsType(), optional=True),
            "prompt_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "transf_log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "encoder_states": NeuralType(('B', 'T', 'D'), ChannelType()),
            "encoder_mask": NeuralType(('B', 'T'), MaskType()),
        }

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        assert config.get("use_lhotse", False), (
            "Multi-task model only supports dataloading with Lhotse. "
            "Please set config.{train,validation,test}_ds.use_lhotse=True"
        )
        global_rank = config.get("global_rank", self.global_rank)
        world_size = config.get("world_size", self.world_size)
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=global_rank,
            world_size=world_size,
            dataset=PromptedTextToTextLhotseDataset(
                tokenizer=self.tokenizer,
                prompt=self.prompt,
            ),
            tokenizer=self.tokenizer,
        )

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        transcript=None,
        transcript_length=None,
    ):
        """
        Forward pass of the model.
        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            # TODO: Add support for `transcript` and `transcript_length` in the docstring

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        enc_states, encoded_len = self.encoder(input_signal), input_signal_length
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)

        transf_log_probs = None
        if transcript is not None:
            dec_mask = lens_to_mask(transcript_length, transcript.shape[1]).to(transcript.dtype)
            dec_states = self.transf_decoder(
                input_ids=transcript, decoder_mask=dec_mask, encoder_embeddings=enc_states, encoder_mask=enc_mask
            )
            transf_log_probs = self.log_softmax(hidden_states=dec_states)

        return transf_log_probs, encoded_len, enc_states, enc_mask

    # PTL-specific methods
    def training_step(self, batch: PromptedTextToTextMiniBatch, batch_nb):

        if batch is None:
            return torch.tensor([0.0])

        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        num_frames = batch.text_lens.sum().float()
        num_tokens = batch.prompted_transcript_lens.sum().float()
        tot_frames = torch.as_tensor(batch.text.numel(), device=num_frames.device, dtype=torch.float)
        tot_tokens = torch.as_tensor(batch.prompted_transcript.numel(), device=num_frames.device, dtype=torch.float)

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.audio,
            input_signal_length=batch.audio_lens,
            transcript=input_ids,
            transcript_length=input_ids_lens,
        )

        # Mask components: 1) discard padding  &  2) discard prompt (notice the negation)
        # For a full decoder sequence O with len M, the loss mask skips the first element,
        # covering the remaining M-1 elements - hence we subtract 1 from prompt lens to account BOS.
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
        else:
            loss_mask = None
        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)

        tensorboard_logs = {
            'train_loss': transf_loss,
            'learning_rate': torch.as_tensor(self._optimizer.param_groups[0]['lr']),
            'batch_size': torch.as_tensor(batch.text.shape[0]),
            'num_frames': num_frames,
            'num_tokens': num_tokens,
            'input_to_padding_ratio': num_frames / tot_frames,
            'output_to_padding_ratio': num_tokens / tot_tokens,
        }

        return {'loss': transf_loss, 'log': tensorboard_logs}

    def validation_pass(self, batch: PromptedAudioToTextMiniBatch, batch_idx, dataloader_idx=0, eval_mode="val"):
        input_ids, labels = batch.get_decoder_inputs_outputs()
        input_ids_lens = batch.prompted_transcript_lens - 1

        transf_log_probs, encoded_len, enc_states, enc_mask = self.forward(
            input_signal=batch.text,
            input_signal_length=batch.text_lens,
            transcript=input_ids,
            transcript_length=batch.prompted_transcript_lens,
        )

        # Mask components: 1) discard padding  &  2) discard prompt (notice the negation)
        # For a full decoder sequence O with len M, the loss mask skips the first element,
        # covering the remaining M-1 elements - hence we subtract 1 from prompt lens to account BOS.
        if self.cfg.get("use_loss_mask_for_prompt", False):
            maxlen = batch.prompted_transcript.shape[1] - 1
            loss_mask = lens_to_mask(input_ids_lens, maxlen) & ~lens_to_mask(batch.prompt_lens - 1, maxlen)
            num_measurements = loss_mask.long().sum()
        else:
            loss_mask = None
            num_measurements = transf_log_probs.shape[0] * transf_log_probs.shape[1]
        transf_loss = self.loss(log_probs=transf_log_probs, labels=labels, output_mask=loss_mask)
        self.val_loss(loss=transf_loss, num_measurements=num_measurements)
        output_dict = {f'{eval_mode}_loss': transf_loss}

        self.wer.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=enc_mask,
            input_ids=batch.prompt,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        output_dict.update({"val_wer": wer, "val_wer_num": wer_num, "val_wer_denom": wer_denom})
        self.wer.reset()

        self.bleu.update(
            predictions=enc_states,
            predictions_lengths=encoded_len,
            targets=batch.transcript,
            targets_lengths=batch.transcript_lens,
            predictions_mask=enc_mask,
            input_ids=batch.prompt,
        )
        bleu_metrics = self.bleu.compute(prefix=f"{eval_mode}_")
        output_dict.update(bleu_metrics)
        self.bleu.reset()

        return output_dict
