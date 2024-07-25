from nemo.collections.common.prompts.formatter import PromptFormatter
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from nemo.collections.asr.modules.transformer import TransformerDecoder, TransformerEncoder
from nemo.core.classes.common import Serialization
from torch import nn
import torch

PROMPT_LOC=["preprocessor", "encoder"] + [str(i) for i in range(0,24)]
PROMPT_METHOD=["concat", "expand", "linear_adapter", "transf_dec_adapter"]

def lens_to_mask(lens, max_length):
    batch_size = lens.shape[0]
    mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
    return mask


class PromptingMixin:
    def _maybe_setup_prompting(self, cfg):
        if cfg.get("prompt_format"):
            self.prompt_format = cfg.prompt_format
            prompt_cls = PromptFormatter.resolve(self.prompt_format)
            self.prompt = prompt_cls(
                tokenizer=self.tokenizer,
                defaults=OmegaConf.to_container(pd) if (pd := cfg.get("prompt_defaults")) is not None else None,
            )
        else:
            self.prompt_format, self.prompt = None, None

    def _maybe_setup_prompt_adapters(self, cfg):
        self.prompt_loc, self.prompt_method = None, None
        self.embed, self.adapter = None, None
        self.num_prompt = 0
        if self.prompt_format:
            assert cfg.get("prompt_cfg"), "Please specify a prompt_cfg"

            self.prompt_loc, self.prompt_method = cfg.prompt_cfg.prompt_loc, cfg.prompt_cfg.prompt_method
            assert self.prompt_loc in PROMPT_LOC and self.prompt_method in PROMPT_METHOD, f"Loc: {self.prompt_loc} and Method: {self.prompt_method} incompatible" 
            
            self.num_prompt = self.cfg.prompt_cfg.num_prompt

            # Sets up embedding
            embed_size = cfg.prompt_cfg.embed_size
            self.embed = nn.Embedding(len(self.tokenizer.special_tokens), embed_size)

            # Set up adapters (if applicable)
            encoded_size= cfg.prompt_cfg.encoded_size
            if self.prompt_method == "linear_adapter":
                self.adapter = nn.Linear(encoded_size + embed_size * self.num_prompt, encoded_size)

            if self.prompt_method == "transf_dec_adapter":
                self.adapter = TransformerDecoder(
                    num_layers=cfg.prompt_cfg.transf_layers,
                    hidden_size=encoded_size,
                    inner_size=cfg.prompt_cfg.transf_hidden,
                    num_attention_heads=cfg.prompt_cfg.transf_heads,
                    pre_ln = True,
                    )


    def apply_prompt(self, encoded, encoded_len, prompt, prompt_len, loc):
        if not self.prompt_loc or loc != self.prompt_loc:
            return encoded, encoded_len
        prompt = self.embed(prompt).transpose(1, 2)
        if self.prompt_method == "concat":
            return torch.cat((prompt, encoded), dim=2), encoded_len + prompt_len

        elif self.prompt_method == "transf_dec_adapter":
            prompt, encoded = prompt.transpose(1,2), encoded.transpose(1,2)
            prompt_mask, encoded_mask = lens_to_mask(prompt_len, prompt.shape[1]).to(prompt.dtype), lens_to_mask(encoded_len, encoded.shape[1]).to(encoded.dtype)
            decoded = self.adapter(
                encoded,
                encoded_mask,
                prompt,
                prompt_mask)
            return decoded.transpose(1,2), encoded_len

        else:
            prompt = prompt.reshape(prompt.shape[0], -1, 1)
            prompt = prompt.expand(-1, -1, encoded.shape[2])
            encoded = torch.cat((encoded, prompt), dim=1)

            if self.prompt_method == "expand":
                return encoded, encoded_len
            
            if self.prompt_method == "linear_adapter":
                encoded = self.adapter(encoded.transpose(1,2)).transpose(1,2)
                return encoded, encoded_len
