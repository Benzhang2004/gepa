# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Shipped pipelines, by name. A pipeline is a module with ``build(stage_opts, defaults, deps) -> Pipeline``."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from gepa.async_stages.pipelines import reflective, sibling_proposals

PIPELINES: dict[str, Callable[..., Any]] = {
    "reflective": reflective.build,
    "sibling_proposals": sibling_proposals.build,
}


def resolve(name: str) -> Callable[..., Any]:
    """A registered pipeline by name, or a dotted module path exposing ``build``."""
    if name in PIPELINES:
        return PIPELINES[name]
    if "." in name:
        module = importlib.import_module(name)
        build = getattr(module, "build", None)
        if build is None:
            raise ValueError(f"module {name!r} has no build()")
        return build
    raise ValueError(f"unknown pipeline {name!r}; registered: {sorted(PIPELINES)}")
