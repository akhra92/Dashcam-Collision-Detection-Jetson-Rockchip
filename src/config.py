"""Lightweight config loader: YAML -> attribute-accessible dict."""
from __future__ import annotations
import yaml
from pathlib import Path


class Cfg(dict):
    """dict that also supports attribute access and nested wrapping."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path: str | Path) -> Cfg:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(raw)
    cfg["_config_path"] = str(path)
    return cfg
