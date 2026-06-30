# Copyright 2026 Bytedance Ltd. and/or its affiliates
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


def _resolve_rwkv_tokenizer_cls(tokenizer_cls: type | None = None) -> type:
    if tokenizer_cls is not None:
        return tokenizer_cls
    from vllm.tokenizers.rwkv import RWKVTokenizer

    return RWKVTokenizer


class PickleableRWKVTokenizer:
    """Pickle-safe proxy for vLLM-RWKV's trie-backed tokenizer."""

    def __init__(
        self,
        tokenizer_path: str | None = None,
        *,
        tokenizer_cls: type | None = None,
        **kwargs: Any,
    ):
        self._tokenizer_path = tokenizer_path
        self._kwargs = dict(kwargs)
        self._tokenizer = _build_native_rwkv_tokenizer(
            tokenizer_path,
            tokenizer_cls=tokenizer_cls,
            **kwargs,
        )
        self.padding_side = getattr(self._tokenizer, "padding_side", "right")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    @property
    def bos_token_id(self) -> int:
        return int(getattr(self._tokenizer, "bos_token_id", 0))

    @property
    def eos_token_id(self) -> int:
        return int(getattr(self._tokenizer, "eos_token_id", 0))

    @property
    def pad_token_id(self) -> int:
        return int(getattr(self._tokenizer, "pad_token_id", self.eos_token_id))

    @property
    def pad_token(self) -> str:
        return getattr(self._tokenizer, "pad_token", "")

    @property
    def vocab_size(self) -> int:
        return int(getattr(self._tokenizer, "vocab_size"))

    def __len__(self) -> int:
        return len(self._tokenizer)

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return list(self._tokenizer.encode(*args, **kwargs))

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: Any | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ):
        """Expose vLLM-RWKV's tokenizer through Verl's HF-style call shape."""
        if tools:
            raise ValueError("RWKV tokenizer chat template does not support tool schemas")
        try:
            output = self._tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        except NotImplementedError:
            output = self._plain_text_prompt(messages)
            if tokenize:
                output = self.encode(output)
        if not return_dict:
            return output
        if not tokenize:
            return {"text": output}
        result = {"input_ids": output, "attention_mask": [1] * len(output)}
        if kwargs.get("return_tensors") == "pt":
            import torch

            result = {key: torch.tensor([value], dtype=torch.long) for key, value in result.items()}
        return result

    def _plain_text_prompt(self, messages) -> str:
        user_parts = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") not in {None, "user"}:
                continue
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            text = self._content_to_text(content)
            if text:
                user_parts.append(text)
        problem = "\n".join(user_parts)
        return f"User: {problem}\n\nAssistant: <think"

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            segments = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        segments.append(str(item.get("text", "")))
                else:
                    segments.append(str(item))
            return "".join(segments)
        return str(content)

    def pad(
        self,
        encoded_inputs: dict[str, Any],
        *,
        padding: str = "max_length",
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
    ) -> dict[str, Any]:
        if padding != "max_length" or max_length is None:
            raise ValueError("RWKV tokenizer pad currently supports padding='max_length' with max_length set")
        input_ids = encoded_inputs["input_ids"]
        single = bool(input_ids and isinstance(input_ids[0], int))
        rows = [input_ids] if single else input_ids
        padded_rows = []
        attention_rows = []
        for row in rows:
            row = list(row)[:max_length]
            pad_size = max_length - len(row)
            pad_values = [self.pad_token_id] * pad_size
            mask_values = [0] * pad_size
            token_mask = [1] * len(row)
            if self.padding_side == "left":
                padded_rows.append(pad_values + row)
                attention_rows.append(mask_values + token_mask)
            else:
                padded_rows.append(row + pad_values)
                attention_rows.append(token_mask + mask_values)
        result = {"input_ids": padded_rows[0] if single else padded_rows}
        if return_attention_mask:
            result["attention_mask"] = attention_rows[0] if single else attention_rows
        if return_tensors == "pt":
            import torch

            result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
        elif return_tensors is not None:
            raise ValueError(f"Unsupported tensor type for RWKV tokenizer pad: {return_tensors!r}")
        return result

    def __getstate__(self) -> dict[str, Any]:
        return {
            "tokenizer_path": self._tokenizer_path,
            "kwargs": self._kwargs,
            "padding_side": self.padding_side,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(
            state["tokenizer_path"],
            **state["kwargs"],
        )
        self.padding_side = state.get("padding_side", "right")


def _build_native_rwkv_tokenizer(
    tokenizer_path: str | None = None,
    *,
    tokenizer_cls: type | None = None,
    **kwargs: Any,
) -> Any:
    cls = _resolve_rwkv_tokenizer_cls(tokenizer_cls)
    if tokenizer_path:
        return cls(tokenizer_path, **kwargs)
    return cls(**kwargs)


def build_rwkv_tokenizer(
    tokenizer_path: str | None = None,
    *,
    tokenizer_cls: type | None = None,
    pickleable: bool = False,
    **kwargs: Any,
) -> Any:
    """Build the native RWKV tokenizer.

    Source: ``vllm.tokenizers.rwkv.RWKVTokenizer`` in the vLLM-RWKV checkout.
    Verl wraps it only to provide the small HF-style surface needed by PPO data
    and reward-loop paths.
    """

    if pickleable:
        return PickleableRWKVTokenizer(
            tokenizer_path,
            tokenizer_cls=tokenizer_cls,
            **kwargs,
        )
    return _build_native_rwkv_tokenizer(
        tokenizer_path,
        tokenizer_cls=tokenizer_cls,
        **kwargs,
    )
