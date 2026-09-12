from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_prompt_bundle(path: str | Path) -> dict[str, Any]:
    prompt_path = Path(path)
    raw = prompt_path.read_bytes()
    config = json.loads(raw)
    base_prompt = config.pop("base_prompt", None)
    if base_prompt:
        base = load_prompt_bundle(prompt_path.parent / str(base_prompt))
        base.pop("sha256", None)
        base.update(config)
        config = base
    required = {"version", "detector_system", "detector_user", "corrector_system", "corrector_user"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"prompt bundle missing fields: {sorted(missing)}")
    detector_passes = config.get("detector_passes", [])
    if not isinstance(detector_passes, list):
        raise ValueError("detector_passes must be a list")
    for index, detector_pass in enumerate(detector_passes):
        if not isinstance(detector_pass, dict):
            raise ValueError(f"detector_passes[{index}] must be an object")
        missing_pass_fields = {"name", "system", "user"} - set(detector_pass)
        if missing_pass_fields:
            raise ValueError(
                f"detector_passes[{index}] missing fields: {sorted(missing_pass_fields)}"
            )
    corrector_passes = config.get("corrector_passes", [])
    if not isinstance(corrector_passes, list):
        raise ValueError("corrector_passes must be a list")
    for index, corrector_pass in enumerate(corrector_passes):
        if not isinstance(corrector_pass, dict):
            raise ValueError(f"corrector_passes[{index}] must be an object")
        missing_pass_fields = {"name", "system", "user"} - set(corrector_pass)
        if missing_pass_fields:
            raise ValueError(
                f"corrector_passes[{index}] missing fields: {sorted(missing_pass_fields)}"
            )
    postprocessors = config.get("deterministic_postprocessors", [])
    if not isinstance(postprocessors, list) or not all(isinstance(name, str) for name in postprocessors):
        raise ValueError("deterministic_postprocessors must be a list of strings")
    # Hash the actual selected file so an inherited ablation remains separately
    # identifiable from its v1.1 parent in manifests and cache purposes.
    config["sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def render(template: str, payload: dict[str, Any]) -> str:
    return template.replace("{{PAYLOAD}}", json.dumps(payload, ensure_ascii=False, indent=2))
