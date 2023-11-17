# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
from math import ceil

import torch
from typing import List
import editdistance
import torch.distributed as dist
from tqdm.auto import tqdm


from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

from nemo.core.classes.mixins import AccessMixin

from sacrebleu import corpus_bleu


__all__ = ['EncDecCTCModelBPE_ST']



class EncDecCTCModelBPE_ST(EncDecCTCModelBPE, ASRBPEMixin):

    @torch.no_grad()
    def translate(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
    ) -> List[str]:
        hypotheses = self.transcribe(paths2audio_files=paths2audio_files, batch_size=batch_size, logprobs=logprobs, return_hypotheses=return_hypotheses)
        return hypotheses

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)
        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            log_probs, encoded_len, predictions = self.forward(
                processed_signal=signal, processed_signal_length=signal_len
            )
        else:
            log_probs, encoded_len, predictions = self.forward(input_signal=signal, input_signal_length=signal_len)

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )
        loss_value, output_dict = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=False, log_wer_num_denom=False, log_prefix="val_",
        )
        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        current_hypotheses, all_hyp = self.decoding.ctc_decoder_predictions_tensor(
            log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
        )
        if all_hyp is None:
            translations = current_hypotheses
        else:
            translations = all_hyp
        output_dict.update({
            'val_loss': loss_value,
            'translations': translations,
            'ground_truths': ground_truths
        })
        return output_dict

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0, eval_mode: str = "val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        for output in outputs:
            eval_loss = torch.stack([x['val_loss'] for x in output]).mean()
            translations = list(itertools.chain(*[x['translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            if self.world_size > 1:
                dist.all_gather_object(
                    tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
                )
            else:
                tr_and_gt[0] = [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']

            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")
                sb_score = sacre_bleu.score * self.world_size

                wer_scores, wer_words = 0, 0
                for h, r in zip(_translations, _ground_truths):
                    wer_words += len(r.split())
                    wer_scores += editdistance.eval(h.split(), r.split())
                wer_score = 1.0 * wer_scores * self.world_size / wer_words

            else:
                sb_score = 0.0
                wer_score = 0.0

            self.log(f"{eval_mode}_loss", eval_loss, sync_dist=True)
            self.log(f"{eval_mode}_sacreBLEU", sb_score, sync_dist=True)
            self.log(f"{eval_mode}_WER", wer_score, sync_dist=True)

