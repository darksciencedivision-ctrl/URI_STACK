from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _now_utc_iso() -> str:
    # good enough for “deterministic-ish”
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_path(p: Any) -> Path:
    if isinstance(p, Path):
        return p
    if p is None:
        return Path(".")
    return Path(str(p))


def _ollama_generate(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    """
    Minimal Ollama /api/generate client.
    Returns dict with: ok(bool), text(str), raw(dict), error(str|None)
    """
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_new_tokens),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            raw = json.loads(body)
            text = raw.get("response", "")
            return {"ok": True, "text": text, "raw": raw, "error": None}
    except Exception as e:
        return {"ok": False, "text": "", "raw": {}, "error": str(e)}


@dataclass
class SovereignConfig:
    # canonical fields
    model: str = "qwen2.5:32b-instruct"
    lattice_path: Path = field(default_factory=lambda: Path(r"C:\ai_control\SOVEREIGN\ui_outbox"))
    thoughts_per_run: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    ppl_threshold: float = 9999.0
    allow_self_modify: bool = False
    allow_vision_auto_train: bool = False
    write_artifacts: bool = True

    # compat sink
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        # ---- COMPAT SHIMS (MANDATORY) ----
        if "model_id" in kwargs and "model" not in kwargs:
            kwargs["model"] = kwargs.pop("model_id")
        if "load_path" in kwargs and "lattice_path" not in kwargs:
            kwargs["lattice_path"] = kwargs.pop("load_path")
        if "threshold_ppl" in kwargs and "ppl_threshold" not in kwargs:
            kwargs["ppl_threshold"] = kwargs.pop("threshold_ppl")

        # auto-train vision variants
        for k in ("auto_train_vision", "auto_train_vision_model", "allow_auto_train_vision", "vision_auto_train"):
            if k in kwargs and "allow_vision_auto_train" not in kwargs:
                kwargs["allow_vision_auto_train"] = kwargs.pop(k)

        # canonical assigns (ignore unknown -> extra)
        fields = {
            "model", "lattice_path", "thoughts_per_run", "max_new_tokens", "temperature", "top_p",
            "ppl_threshold", "allow_self_modify", "allow_vision_auto_train", "write_artifacts"
        }

        # defaults
        object.__setattr__(self, "model", str(kwargs.pop("model", "llama3.1:latest")))
        object.__setattr__(self, "lattice_path", _coerce_path(kwargs.pop("lattice_path", Path(r"C:\ai_control\SOVEREIGN\ui_outbox"))))
        object.__setattr__(self, "thoughts_per_run", int(kwargs.pop("thoughts_per_run", 1)))
        object.__setattr__(self, "max_new_tokens", int(kwargs.pop("max_new_tokens", 512)))
        object.__setattr__(self, "temperature", float(kwargs.pop("temperature", 0.7)))
        object.__setattr__(self, "top_p", float(kwargs.pop("top_p", 0.9)))
        object.__setattr__(self, "ppl_threshold", float(kwargs.pop("ppl_threshold", 9999.0)))
        object.__setattr__(self, "allow_self_modify", bool(kwargs.pop("allow_self_modify", False)))
        object.__setattr__(self, "allow_vision_auto_train", bool(kwargs.pop("allow_vision_auto_train", False)))
        object.__setattr__(self, "write_artifacts", bool(kwargs.pop("write_artifacts", True)))

        extra = {}
        for k, v in list(kwargs.items()):
            if k not in fields:
                extra[k] = v
        object.__setattr__(self, "extra", extra)


@dataclass
class OracleConfig:
    model: str = "llama3.1:latest"
    ppl_threshold: float = 9999.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any):
        if "model_id" in kwargs and "model" not in kwargs:
            kwargs["model"] = kwargs.pop("model_id")
        if "threshold_ppl" in kwargs and "ppl_threshold" not in kwargs:
            kwargs["ppl_threshold"] = kwargs.pop("threshold_ppl")

        object.__setattr__(self, "model", str(kwargs.pop("model", "llama3.1:latest")))
        object.__setattr__(self, "ppl_threshold", float(kwargs.pop("ppl_threshold", 9999.0)))

        object.__setattr__(self, "extra", dict(kwargs))


class Oracle:
    """
    Gatekeeper. Since we’re not computing real perplexity here,
    we enforce *shape* + basic quality heuristics to stop junk.
    """

    def __init__(self, config: Optional[OracleConfig] = None):
        self.config = config or OracleConfig()

    def review(self, text: str) -> Dict[str, Any]:
        warnings = []
        t = (text or "").strip()

        if not t:
            warnings.append({"code": "EMPTY", "message": "Model returned empty output."})

        # crude “slurry detector”
        code_fence_count = t.count("```")
        if code_fence_count >= 6:
            warnings.append({"code": "CODE_SLURRY", "message": "Excessive fenced code blocks; likely dumping."})

        if len(t) < 30:
            warnings.append({"code": "TOO_SHORT", "message": "Output suspiciously short."})

        approved = len(warnings) == 0
        return {"approved": approved, "warnings": warnings}


class Sovereign:
    """
    Lightweight cycle runner for the Streamlit UI.
    UI calls this, this writes minimal artifacts, no hidden state.
    """

    def __init__(self, model: str, lattice_path: str | Path, config: SovereignConfig):
        self.model = model
        self.lattice_path = _coerce_path(lattice_path)
        self.config = config
        self.oracle = Oracle(OracleConfig(model=self.model, ppl_threshold=self.config.ppl_threshold))

        self.lattice_path.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, ollama_host: str = "http://localhost:11434") -> Dict[str, Any]:
        ts = _now_utc_iso()
        raw = _ollama_generate(
            prompt=prompt,
            model=self.model,
            host=ollama_host,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )
        text = raw["text"] if raw["ok"] else ""
        gate = self.oracle.review(text)

        out = {
            "timestamp_utc": ts,
            "model": self.model,
            "prompt": prompt,
            "ok": bool(raw["ok"]),
            "error": raw["error"],
            "approved": bool(gate["approved"]),
            "warnings": gate["warnings"],
            "text": text,
        }

        if self.config.write_artifacts:
            # keep it simple and explicit
            (self.lattice_path / "last_run.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
            (self.lattice_path / "last_text.txt").write_text(text or "", encoding="utf-8")

        return out

    def maybe_train(self, approved: bool, text: str) -> Dict[str, Any]:
        # Placeholder. You asked for the hook, not the fantasy.
        if not approved or not self.config.allow_vision_auto_train:
            return {"trained": False, "reason": "disabled_or_not_approved"}
        return {"trained": False, "reason": "not_implemented"}

