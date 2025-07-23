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

from typing import Any

import torch
from lhotse import MonoCut
from lhotse.cut import Cut, MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter, _mangled
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY_BOS,
    CANARY_EOS,
    CANARY_SPECIAL_TOKENIZER,
    CanaryTokenizer,
)


class CanaryRivaPromptFormatter(PromptFormatter):
    NAME = "canary_riva"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY_BOS}|target_lang|",
            "slots": {
                "target_lang": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"|source_lang| |text|{CANARY_EOS}",
            "slots": {
                "source_lang": Modality.Text,
                "text": Modality.Text,
            },
        },
    }


    def encode_turn(
        self, prompt_template: str, expected_slots: dict[str, Modality], slot_values: dict[str, Any]
    ) -> list[int]:
        slot_values = map_manifest_values_to_special_tokens(slot_values)
        prompt = prompt_template
        for slot in expected_slots:
            # For the final substitution of 'slot' in the template we have to mangle it to '|slot|' anyway,
            # but 'slot' form enables to use valid python identifiers as **kwargs
            # for passing slots around in user functions.
            value = slot_values.get(slot)
            assert value is not None, f"Missing required {slot=} in {slot_values=} for {prompt_template=}"
            prompt = prompt.replace(_mangled(slot), value)
        if "text" in expected_slots:
            prompt, txt = prompt.split(" ", 1)
            return self._apply_tokenizer(prompt, lang=slot_values.get(self.PROMPT_LANGUAGE_SLOT)) + self._apply_tokenizer(txt, lang=slot_values["text_language"])
        return self._apply_tokenizer(prompt, lang=slot_values.get(self.PROMPT_LANGUAGE_SLOT))


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()

    any_special_token_present = False

    for k in ("source_lang", "target_lang"):
        if k in slot_values and not ((v := slot_values[k]).startswith("<|") and v.endswith("|>")):
            slot_values[k] = "<|" + slot_values[k] + "|>"
            any_special_token_present = True
    
    # Auto-inject which tokenizer to look up in CanaryTokenizer if not provided,
    # and slots for this turn correspond to user prompt.
    if any_special_token_present and PromptFormatter.PROMPT_LANGUAGE_SLOT not in slot_values:
        slot_values[PromptFormatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER

    return slot_values


@registered_prompt_format_fn(Cut, CanaryRivaPromptFormatter)
def canary_riva(cut: Cut, prompt: CanaryRivaPromptFormatter) -> dict[str, torch.Tensor]:
    """
    Prepend and append control tokens to the token sequence as per Canary format.

    We use the following special tokens:
    * <|startoftranscript|>
    * <|transcribe|>
    * <|translate|>
    * <|nopnc|>
    * <|pnc|>
    * <|endoftext|>
    * <|LANG|> - for each supported language.
    * <|nospeech|>

    The prompt format syntax is as follows:

        <|startoftranscript|> [ <|nospeech|> | <|LANG|> [ <|transcribe|> | <|translate|> ] <|LANG|> [ <|pnc|> | <|nopnc|> ] TEXT <|endoftext|> ]

    Where expression ``[ a | b ]`` denotes expression ``a`` or expression ``b``, and can be nested.
    Note that ``<|LANG|>`` appears twice: the first occurrence is for the "source" language
    (i.e., spoken language in the recording) and the second occurrence is for the "target" language
    (i.e., the language in which we are going to output the text).
    """
    if isinstance(cut, MixedCut):
        cut = cut._first_non_padding_cut
    if not isinstance(cut, MonoCut):
        raise TypeError(
            f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
        )

    # first, validate the utterance
    expected_slots = set(prompt.get_slots("user"))
    missing_keys = expected_slots - set(cut.custom)
    if missing_keys:
        raise RuntimeError(
            f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
            f"Please ensure that every utterance in the input manifests contains these keys."
        )

    turns = [
        dict(
            role="user",
            slots={
                "target_lang": "asr" if cut.custom["source_lang"] == cut.custom["target_lang"] else cut.custom["target_lang"],
                prompt.PROMPT_LANGUAGE_SLOT: CANARY_SPECIAL_TOKENIZER,
            },
        )
    ]
    # If data has no transcript, create empty response with <eos> only.
    text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
    turns.append(
        dict(
            role="assistant",
            slots={
                "text": text,
                "source_lang": cut.custom["source_lang"],
                "text_language":  ifnone(cut.supervisions[0].language, cut.custom.get("target_lang")),
                prompt.PROMPT_LANGUAGE_SLOT: CANARY_SPECIAL_TOKENIZER,
            },
        ),
    )

    ans = prompt.encode_dialog(turns)
    if isinstance(prompt.tokenizer, CanaryTokenizer):
        eos = prompt.tokenizer.eos
    else:  # SPE
        eos = prompt.tokenizer.token_to_id(CANARY_EOS)
    assert eos > -1, f"Invalid tokenizer: tokenizer.token_to_id('{CANARY_EOS}') returned {eos}"
    assert (
        ans["answer_ids"][-1].item() == prompt.tokenizer.eos
    ), f"Expected the last token in answer_ids to be EOS, but we got {ans['answer_ids']}"
    ans["answer_ids"] = ans["answer_ids"][:-1]  # Strip Canary's EOS
    return ans
