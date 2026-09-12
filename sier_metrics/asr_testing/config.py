from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # Keep list/status commands usable before the full ASR env is installed.
    yaml = None


TESTING_ROOT = Path(__file__).resolve().parents[2]
ASR_FULL_ROOT = TESTING_ROOT / "reports" / "asr_full"
ASR_FULL_PREDICTIONS = ASR_FULL_ROOT / "predictions"
ASR_FULL_RUNS = ASR_FULL_ROOT / "runs"
ASR_FULL_METRICS = ASR_FULL_ROOT / "metrics"
DEFAULT_MODELS_CONFIG = TESTING_ROOT / "configs" / "models.yaml"
DEFAULT_HEALTHCHECK_CONFIG = TESTING_ROOT / "configs" / "healthcheck.yaml"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    enabled: bool
    adapter: str
    local_dir: Path
    source: dict[str, Any]
    load: dict[str, Any]
    raw: dict[str, Any]

    @property
    def repo_id(self) -> str | None:
        value = self.source.get("repo_id")
        return str(value) if value else None

    @property
    def revision(self) -> str | None:
        value = self.source.get("revision")
        return str(value) if value else None


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    escaped = False
    out: list[str] = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char in {"'", '"'}:
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
            out.append(char)
            continue
        if char == "#" and in_quote is None:
            break
        out.append(char)
    return "".join(out).rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this directory's configs.

    This fallback intentionally supports only nested mappings, simple lists, comments,
    and scalar values. Install PyYAML for full YAML support.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()
    for index, original in enumerate(lines):
        cleaned = _strip_comment(original)
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        stripped = cleaned.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Fallback YAML parser expected list parent on line {index + 1}: {original}")
            parent.append(_parse_scalar(stripped[2:]))
            continue
        if ":" not in stripped:
            raise ValueError(f"Fallback YAML parser cannot parse line {index + 1}: {original}")
        key, value = stripped.split(":", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"Fallback YAML parser expected mapping parent on line {index + 1}: {original}")
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_container: Any = {}
        for future in lines[index + 1 :]:
            future_clean = _strip_comment(future)
            if not future_clean.strip():
                continue
            future_indent = len(future_clean) - len(future_clean.lstrip(" "))
            if future_indent <= indent:
                break
            if future_clean.strip().startswith("- "):
                next_container = []
            break
        parent[key] = next_container
        stack.append((indent, next_container))
    return root


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _minimal_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_model_specs(config_path: Path = DEFAULT_MODELS_CONFIG, root: Path = TESTING_ROOT) -> dict[str, ModelSpec]:
    data = read_yaml(config_path)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"models must be a mapping in {config_path}")

    specs: dict[str, ModelSpec] = {}
    for key, raw in models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Model entry must be a mapping: {key}")
        local_dir = Path(str(raw.get("local_dir", f"models/{key}")))
        if not local_dir.is_absolute():
            local_dir = root / local_dir
        specs[str(key)] = ModelSpec(
            key=str(key),
            display_name=str(raw.get("display_name", key)),
            enabled=bool(raw.get("enabled", True)),
            adapter=str(raw.get("adapter", key)),
            local_dir=local_dir,
            source=dict(raw.get("source") or {}),
            load=dict(raw.get("load") or {}),
            raw=raw,
        )
    return specs


def load_healthcheck_config(path: Path = DEFAULT_HEALTHCHECK_CONFIG) -> dict[str, Any]:
    return read_yaml(path)


def select_model_specs(
    specs: dict[str, ModelSpec],
    model: str | None = None,
    all_models: bool = False,
) -> list[ModelSpec]:
    if model and all_models:
        raise ValueError("Use either --model or --all, not both")
    if model:
        if model not in specs:
            available = ", ".join(sorted(specs))
            raise KeyError(f"Unknown model '{model}'. Available models: {available}")
        return [specs[model]]
    if all_models:
        return list(specs.values())
    return list(specs.values())


def rel_to_root(path: Path, root: Path = TESTING_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
