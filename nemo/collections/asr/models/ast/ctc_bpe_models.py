# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType, EncodedRepresentation
from nemo.collections.common.parts import transformer_weights_init

__all__ = ['EncDecCTCModelAST']


def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask

class EncDecCTCModelAST(EncDecCTCModelBPE):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    def __init__(self, cfg: DictConfig, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.insertion_method = self.cfg.get("embed_method")
        self.insertion_location = self.cfg.get("embed_loc")

        # Transformer
        self.has_transformer = False
        if cfg.get('transformer'):
            self.has_transformer = True
            self.transformer = EncDecCTCModelAST.from_config_dict(cfg.get('transformer'))
            std_init_range = 1 / cfg.get('transformer').hidden_size ** 0.5
            self.transformer.apply(lambda module: transformer_weights_init(module, std_init_range))


        self.insertion_proj = None
        lang_embed_dim = self.cfg.get("lang_embed_dim")
        if self.insertion_location == "encoder":
            encoded_dim = self.cfg.preprocessor["features"]
            if self.insertion_method == "expand":
                self.insertion_proj = torch.nn.Linear(lang_embed_dim + encoded_dim, encoded_dim)
            else:
                lang_embed_dim = encoded_dim
        elif self.insertion_location in [l for l in range(self.encoder.n_layers)]:
            if self.insertion_method in ["add", "concat"]:
                lang_embed_dim = self.cfg.encoder["d_model"]
            with open_dict(self.cfg):
                self.cfg["lang_embed_dim"] = lang_embed_dim
                self.encoder._update_insertion(self.cfg)
        elif self.insertion_location == "transformer":
            encoded_dim = self.cfg.encoder["d_model"]
            if self.insertion_method == "expand":
                self.insertion_proj = torch.nn.Linear(lang_embed_dim + encoded_dim, encoded_dim)
            else:
                lang_embed_dim = encoded_dim
        elif self.insertion_location == "decoder":
            encoded_dim = self.cfg.decoder["feat_in"]
            if self.insertion_method == "expand":
                self.insertion_proj = torch.nn.Linear(lang_embed_dim + encoded_dim, encoded_dim)
            else:
                lang_embed_dim = encoded_dim
        else:
            raise NameError
        self.lang_embedding = torch.nn.Embedding(
                num_embeddings=len(self.cfg.get("lang_labels")), 
                embedding_dim=lang_embed_dim,
            )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        # Making sure lang_labels align with tokenizers
        if self.tokenizer_type == "agg":
            with open_dict(self.cfg):
                self.cfg.lang_labels = self.tokenizer.langs
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='lang_labels')
        return super()._setup_dataloader_from_config(config)

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None,
        lang_id=None
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
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
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
        lang_vect = self.lang_embedding(lang_id)

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        if self.insertion_location == "encoder":
            processed_signal, processed_signal_length = self._add_embedding(processed_signal, processed_signal_length, lang_vect)

        # Look into mixins/adapter for inter mixin
        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length, lang_id=lang_vect)
        encoded, encoded_len = encoder_output[0], encoder_output[1]

        if self.has_transformer:
            if self.insertion_location == "transformer":
                encoded, encoded_len = self._add_embedding(encoded, encoded_len, lang_vect)
            encoded = encoded.transpose(1, 2)
            mask_encoded = lens_to_mask(encoded_len, encoded.shape[1])
            encoded = self.transformer(encoded, mask_encoded).transpose(1,2)

        if self.insertion_location == "decoder":
            encoded, encoded_len = self._add_embedding(encoded, encoded_len, lang_vect)

        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            'lang_id': NeuralType(tuple('B'), EncodedRepresentation(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, lang_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len, lang_id=lang_id)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        # Add auxiliary losses, if registered
        loss_value = self.add_auxiliary_losses(loss_value)
        # only computing metric when requested in the logs (same as done for final-layer metric below)
        loss_value, tensorboard_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=((batch_nb + 1) % log_every_n_steps == 0)
        )

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        tensorboard_logs.update(
            {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }
        )

        if (batch_nb + 1) % log_every_n_steps == 0:
            self.wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            metrics = self.wer.compute(return_all_metrics=False, prefix="training_batch_")
            self.wer.reset()

            tensorboard_logs.update(metrics)

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, lang_id = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len, lang_id=lang_id)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, interctc_metrics = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=True, log_wer_num_denom=True, log_prefix="val_",
        )
        interctc_metrics.update({'val_loss': loss_value})

        self.wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        metrics = self.wer.compute(prefix="val_")
        self.wer.reset()

        metrics.update(interctc_metrics)
        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)
        return metrics

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    @property
    def bleu(self):
        return self._bleu

    @bleu.setter
    def bleu(self, bleu):
        self._bleu = bleu

    def _add_embedding(self, audio, audio_len, lang):
        # Assumes audio is BxDxT
        input, input_len, lang = audio, audio_len, lang.unsqueeze(-1)

        if self.insertion_method == "concat":
            input = torch.concat([lang, audio], dim=2)
            input_len += 1
        else:
            mask = ~lens_to_mask(input_len, input.shape[2])
            if self.insertion_method == "expand":
                lang = lang.expand(-1, -1, audio.shape[-1])
                lang = lang.masked_fill(mask.unsqueeze(1), 0.0)
                input = torch.concat([audio, lang], dim=1)
                input = self.insertion_proj(input.transpose(1,2)).transpose(1,2)
            elif self.insertion_method == "add":
                lang = lang.expand_as(input)
                lang = lang.masked_fill(mask.unsqueeze(1), 0.0)
                input = input + lang
            else:
                raise NameError
        return input, input_len
    
def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask