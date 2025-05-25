from nemo.collections.asr.metrics import *
from omegaconf import DictConfig
from nemo.core.classes import Serialization
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.prompts.canary import TASK_TRANSCRIBE, TASK_TRANSLATE
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_BOS
from torchmetrics import Metric

from omegaconf import OmegaConf
from collections import defaultdict
from lhotse import CutSet

from typing import Literal, Optional, Sequence, TypeAlias, Union, Dict
import torch 

__all__ = ['MultiTaskMetric']


SUPPORTED_METRICS = {
    "wer": WER,
    "bleu": BLEU,
}

class MultiTaskMetric():
    # trick from torch metrics `SacreBLEUTokenizer`
    _INDEX_FN = {
        "canary": "_canary_index_fn",
        "canary2": "_canary2_index_fn",
    }
    
    def __init__(self, decoding: AbstractMultiTaskDecoding, prompt: PromptFormatter, cfg: DictConfig, log_prediction: bool = True):        
        cfg = OmegaConf.to_container(cfg)

        # Setup prompt validation functiosn.
        assert prompt.NAME in self._INDEX_FN, f"MultiTaskMetric logging is only supported for f{[k for k in self._INDEX_FN.keys()]}"
        self.split_task_indices = getattr(self, f"{self._INDEX_FN[prompt.NAME]}")

        # Setup metric dict
        self._metric_dict, self._slot_dict = {}, {}
        for metric in cfg.pop("metrics"):
            print(metric)
            slots = metric["slots"]
            # TODO: Expand slot coverage as metrics demands.
            assert "task" in slots and len(slots) == 1, "MultiTask metric currently only supports task based constraints. Check 'MultiTaskMetric' cfg."

            # All other elements of config are assumed to be global attributes across metrics.
            metric["log_prediction"] = log_prediction
            metric["compute_on_cpu"] = False
            metric["sync_on_compute"] = False
            metric["dist_sync_on_step"] = True
            metric["full_state_update"] = True
            #metric["decoding"] = decoding
            for k, v in cfg.items():
                if k not in metric:  # do not override explicit metric values.
                    metric[k] = v

            # instantiate metric and constraints
            print(metric["name"])
            self._metric_dict[metric["name"]] = SUPPORTED_METRICS[metric["name"]](decoding, **metric)
            print(self._metric_dict[metric["name"]].process_group)

            self._slot_dict[metric["name"]] = {**slots}

    # TODO: If constraints supported, add properties to `Canary` to simplly return `task` idx.
    def _canary_index_fn(self, prompt_ids: torch.Tensor, slots: Dict[str, (str | bool)]) -> torch.Tensor:
        if slots["task"] in TASK_TRANSLATE:
            # 1 -> `source_lang` in canary, 3 -> 'target_lang. Use these instead of task ID to avoid lookup of task id.
            condition_met = prompt_ids[:,1] != prompt_ids[:,3]
        else:  # default to transcribe
            condition_met = prompt_ids[:,1] == prompt_ids[:,3]
        print(condition_met.device)
        return torch.nonzero(condition_met, as_tuple=False).squeeze()
    
    # TODO: If constraints supported, add properties to `Canary2` to simplly return `task` idx.
    # Note: This is offset by decoder context.
    def _canary2_index_fn(self, prompt_ids: torch.Tensor, slots: Dict[str, str | bool]) -> torch.Tensor:
        bos = prompt_ids == self.prompt.tokenizer.bos_id
        bos_idx = torch.nonzero(bos, tuple=False).squeeze()
        if slots["task"] in TASK_TRANSLATE:
            condition_met = prompt_ids[:, bos_idx + 2] != prompt_ids[:, bos_idx + 3]   # 2 -> `source_lang` in canary, 3 -> 'target_lang`
        else:  # default to transcribe
            condition_met = prompt_ids[:, bos_idx + 2] == prompt_ids[:, bos_idx + 3]
        return torch.nonzero(condition_met, as_tuple=False).squeeze()

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        predictions_mask: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        cuts: Optional[CutSet] = None,
    ):
        print([c for c in cuts])
        for name, slot in self._slot_dict.items():
            indices = self.split_task_indices(input_ids, slot)
            if indices.numel() == 0:  # for if the metric case is niche.
                continue
            print(indices)
            metric = self._metric_dict[name]
            print(metric.dist_sync_on_step)
            metric.update(
                predictions=predictions[indices],
                predictions_lengths=predictions_lengths[indices],
                predictions_mask=predictions_mask[indices],
                targets=targets[indices],
                targets_lengths=targets_lengths[indices],
                input_ids=input_ids[indices],
                #cuts=cuts[indices] if cuts else None,
            )
                
    def compute(self, return_all_metrics=True, prefix="", suffix=""):
        output_dict = {}
        for metric in self._slot_dict:
            output_dict.update(
                self._metric_dict[metric].compute(
                    return_all_metrics=return_all_metrics,
                    prefix=prefix,
                    suffix=suffix,
                )
            )
        return output_dict

    def reset(self):
        for metric in self._slot_dict:
            self._metric_dict[metric].reset()
