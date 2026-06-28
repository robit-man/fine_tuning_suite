"""Training Suite package.

The original scripts remain importable as compatibility entrypoints. New code
is organized under package modules so the CLI and web dashboard can share the
same inspection, job, export, and evaluation logic.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
