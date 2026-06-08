"""Shared utilities for the figure_generation scripts.

Each `generate_*.py` is expected to call `lib.style.apply()` before creating
any matplotlib figures, and to route its final PDF through `lib.io.save()`.
"""
from . import data, hull, io, plot, style  # noqa: F401
