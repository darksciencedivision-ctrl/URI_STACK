from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def _normalize_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("caption", "answer", "response", "text"):
            if k in obj:
                return _normalize_text(obj[k])
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return "".join(_normalize_text(x) for x in obj)
    try:
        if isinstance(obj, Iterable):
            return "".join(_normalize_text(x) for x in obj)
    except TypeError:
        pass
    return str(obj)


class SovereignVLM:
    def __init__(self, model_id="vikhyatk/moondream2", revision="2025-01-09"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eyes = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def see_bytes(self, image_bytes: bytes, prompt: str | None = None) -> str:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        if prompt and hasattr(self.eyes, "encode_image"):
            with torch.no_grad():
                enc = self.eyes.encode_image(img)
                out = self.eyes.answer_question(enc, prompt, self.tokenizer)
                return _normalize_text(out)

        with torch.no_grad():
            out = self.eyes.caption(img, stream=True)
        return _normalize_text(out)

