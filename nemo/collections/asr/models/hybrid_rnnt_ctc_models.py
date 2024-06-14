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

import copy
from typing import Any, List, Optional, Tuple
import random
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm
from math import exp
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics import WER, BLEU
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.utils import logging, model_utils
from nemo.collections.common.parts import transformer_weights_init


class EncDecHybridRNNTCTCModel(EncDecRNNTModel, ASRBPEMixin):
    """Base class for hybrid RNNT/CTC models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        self.rnnt_losses = torch.tensor(10, device=self.device, requires_grad=False)
        self.rnnt_wer_log = torch.ones(1, device=self.device, requires_grad=False)

        self.ctc_losses = torch.tensor(10, device=self.device, requires_grad=False)
        self.ctc_wer_log = torch.ones(1, device=self.device, requires_grad=False)

        self.ctc_drop = self.cfg.get("ctc_drop")
        self.anneal = cfg.anneal

        transf_encoder_cfg_dict = self.cfg.get('transf_encoder', None)
        # Whether to add Transformer Encoder block between Conformer and Transformer Decoder
        self.use_transf_encoder = False
        if transf_encoder_cfg_dict is not None and transf_encoder_cfg_dict['num_layers'] > 0:
            self.transf_encoder = EncDecHybridRNNTCTCModel.from_config_dict(transf_encoder_cfg_dict)
            # Initialize weights
            std_init_range = 1 / transf_encoder_cfg_dict.hidden_size ** 0.5
            self.transf_encoder.apply(lambda module: transformer_weights_init(module, std_init_range))

        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )
        with open_dict(self.cfg.aux_ctc):
            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = EncDecRNNTModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

    @torch.no_grad()
    def transcribe(
        self,
        audio: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            audio: (a list) of paths to audio files. \
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            logprobs: (bool) whether to return ctc logits insted of hypotheses

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        if self.cur_decoder not in ["ctc", "rnnt"]:
            raise ValueError(
                f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
            )

        return super().transcribe(
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            override_config=override_config,
        )

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.unfreeze()

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        if self.cur_decoder == "rnnt":
            return super()._transcribe_forward(batch, trcfg)

        # CTC Path
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])

        logits = self.ctc_decoder(encoder_output=encoded)
        output = dict(logits=logits, encoded_len=encoded_len)

        del encoded
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Tuple[List['Hypothesis'], List['Hypothesis']]:
        if self.cur_decoder == "rnnt":
            return super()._transcribe_output_processing(outputs, trcfg)

        # CTC Path
        logits = outputs.pop('logits')
        encoded_len = outputs.pop('encoded_len')

        best_hyp, all_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
            logits, encoded_len, return_hypotheses=trcfg.return_hypotheses,
        )
        logits = logits.cpu()

        if trcfg.return_hypotheses:
            # dump log probs per file
            for idx in range(logits.shape[0]):
                best_hyp[idx].y_sequence = logits[idx][: encoded_len[idx]]
                if best_hyp[idx].alignments is None:
                    best_hyp[idx].alignments = best_hyp[idx].y_sequence

        # DEPRECATED?
        # if logprobs:
        #     for logit, elen in zip(logits, encoded_len):
        #         logits_list.append(logit[:elen])

        del logits, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp

        return (hypotheses, all_hypotheses)

    def change_vocabulary(
        self,
        new_vocabulary: List[str],
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for CTC decoding, which is optional and can be used to change decoding type.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            if self.ctc_decoder.vocabulary == new_vocabulary:
                logging.warning(
                    f"Old {self.ctc_decoder.vocabulary} and new {new_vocabulary} match. Not changing anything."
                )
            else:
                if new_vocabulary is None or len(new_vocabulary) == 0:
                    raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
                decoder_config = self.ctc_decoder.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)
                new_decoder_config['vocabulary'] = new_vocabulary
                new_decoder_config['num_classes'] = len(new_vocabulary)

                del self.ctc_decoder
                self.ctc_decoder = EncDecHybridRNNTCTCModel.from_config_dict(new_decoder_config)
                del self.ctc_loss
                self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
                )

                if ctc_decoding_cfg is None:
                    # Assume same decoding config as before
                    logging.info("No `ctc_decoding_cfg` passed when changing decoding strategy, using internal config")
                    ctc_decoding_cfg = self.cfg.aux_ctc.decoding

                # Assert the decoding config with all hyper parameters
                ctc_decoding_cls = OmegaConf.structured(CTCDecodingConfig)
                ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
                ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

                self.ctc_decoding = CTCDecoding(decoding_cfg=ctc_decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

                self.ctc_wer = WER(
                    decoding=self.ctc_decoding,
                    use_cer=self.ctc_wer.use_cer,
                    log_prediction=self.ctc_wer.log_prediction,
                    dist_sync_on_step=True,
                )

                # Update config
                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg

                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoder = new_decoder_config

                ds_keys = ['train_ds', 'validation_ds', 'test_ds']
                for key in ds_keys:
                    if key in self.cfg:
                        with open_dict(self.cfg[key]):
                            self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

                logging.info(f"Changed the tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            self.cur_decoder = "rnnt"
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.aux_ctc.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.cur_decoder = "ctc"
        logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }
        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb
        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False        

        drop_ctc = False
        if self.anneal:
            prob = (1 - (self.ctc_drop / (self.ctc_drop + exp(self.current_epoch / self.ctc_drop))))
            drop_ctc = prob > random.random()
        else:
            drop_ctc = self.ctc_drop > random.random()
        if drop_ctc:
            self.encoder.freeze()

        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if drop_ctc:
            # During training, loss must be computed, so decoder forward is necessary
            decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

            enc_states = encoded.permute(0, 2, 1)
            enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
            encoded = self.transf_encoder(enc_states, enc_mask).transpose(1, 2)

            # Fused joint step
            joint = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                autoregressive_inference=False,
            )

            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )
            self.rnnt_losses = loss_value.detach()

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                wer, _, _ = self.wer.compute()
                self.rnnt_wer_log = wer
                self.wer.reset()

        else:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            loss_value = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            self.ctc_losses = loss_value.detach()

            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer_log = wer
                self.ctc_wer.reset()

        tensorboard_logs["train_ctc_loss"] = self.ctc_losses
        tensorboard_logs["train_ctc_wer"] = self.ctc_wer_log

        tensorboard_logs["train_rnnt_loss"] = self.rnnt_losses
        tensorboard_logs["train_rnnt_wer"] = self.rnnt_wer_log

        # Log items
        self.log_dict(tensorboard_logs)
        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        self.encoder.unfreeze()
        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        tensorboard_logs = {}
        loss_value = None

        signal, signal_len, transcript, transcript_len = batch
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        ## CTC
        log_probs = self.ctc_decoder(encoder_output=encoded)
        self.ctc_wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer
        self.ctc_wer.reset()

        self.ctc_bleu.update(
            predictions=log_probs,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len
        )
        bleu_metrics = self.ctc_bleu.compute(prefix=f"val_", suffix="_ctc")
        tensorboard_logs.update(bleu_metrics)
        self.ctc_bleu.reset()

        # TDT
        enc_states = encoded.permute(0, 2, 1)
        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1]).to(enc_states.dtype)
        encoded = self.transf_encoder(enc_states, enc_mask).transpose(1, 2)

        self.wer.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len,
        )
        wer, wer_num, wer_denom = self.wer.compute()
        tensorboard_logs['val_wer_num'] = wer_num
        tensorboard_logs['val_wer_denom'] = wer_denom
        tensorboard_logs['val_wer'] = wer
        self.wer.reset()

        self.bleu.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len
        )
        bleu_metrics = self.bleu.compute(prefix="val_")
        tensorboard_logs.update(bleu_metrics)
        self.bleu.reset()

        if self.compute_eval_loss:
            decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            joint = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                autoregressive_inference=False,
            )
            loss_value = self.loss(log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            tensorboard_logs['val_loss'] = loss_value

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}

        ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
        ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
        tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        bleu_pred_len = torch.stack([x[f"val_bleu_pred_len"] for x in outputs]).sum()
        bleu_target_len = torch.stack([x[f"val_bleu_target_len"] for x in outputs]).sum()
        bleu_num = torch.stack([x[f"val_bleu_num"] for x in outputs]).sum(dim=0)
        bleu_denom = torch.stack([x[f"val_bleu_denom"] for x in outputs]).sum(dim=0)
        tensorboard_logs["val_bleu"] = self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)

        ctc_bleu_pred_len = torch.stack([x[f"val_bleu_pred_len_ctc"] for x in outputs]).sum()
        ctc_bleu_target_len = torch.stack([x[f"val_bleu_target_len_ctc"] for x in outputs]).sum()
        ctc_bleu_num = torch.stack([x[f"val_bleu_num_ctc"] for x in outputs]).sum(dim=0)
        ctc_bleu_denom = torch.stack([x[f"val_bleu_denom_ctc"] for x in outputs]).sum(dim=0)
        tensorboard_logs["val_bleu_ctc"] = self.bleu._compute_bleu(ctc_bleu_pred_len, ctc_bleu_target_len, ctc_bleu_num, ctc_bleu_denom)

        metrics = {**val_loss_log, 'log': tensorboard_logs}
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['test_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['test_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['test_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**test_loss_log, 'log': tensorboard_logs}
        return metrics

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        if self.cur_decoder == 'rnnt':
            return ['encoder', 'decoder_joint']
        else:
            return ['self']

    @property
    def output_module(self):
        if self.cur_decoder == 'rnnt':
            return self.decoder
        else:
            return self.ctc_decoder

def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask
