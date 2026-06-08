"""Data loading and filter helpers.

The JSON files are flat lists of run-records; every filter in the figures is
expressible as a predicate over a record's fields. This module provides

* three `load_*` functions for the JSON files,
* small per-run-type filter helpers that return concrete tuples/lists of
  primitives (no nested `run_info` dicts — callers can use them directly),
* the constants `TOKEN_LIST`, `PT_LR`, `TOKEN_PER_PARAM_MAP` that the paper
  uses to pin down experiments.

Float equality for `rho` is tolerant to 1e-9 (SAM ρ values include 0.05, 0.02,
0.15, etc. stored as Python floats).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"

# --- constants -------------------------------------------------------------

TOKEN_LIST = {
    20:  [4, 8, 16, 32, 64],
    60:  [12, 24, 48, 96, 192],
    150: [15, 30, 60, 120, 240],
}

PT_LR = {
    20:  {4: 3e-3, 8: 3e-3, 16: 3e-3, 32: 1e-3, 64: 1e-3},
    60:  {12: 1e-3, 24: 1e-3, 48: 6e-4, 96: 6e-4, 192: 3e-4},
    150: {15: 1e-3, 30: 1e-3, 60: 6e-4, 120: 3e-4, 240: 3e-4},
}

TOKEN_PER_PARAM_MAP = {
    100:  {150: 15},
    200:  {20: 4,  60: 12, 150: 30},
    400:  {20: 8,  60: 24, 150: 60},
    800:  {20: 16, 60: 48, 150: 120},
    1600: {20: 32, 60: 96, 150: 240},
    3200: {20: 64, 60: 192},
}


# Per-figure scatter window, reproduced from the old pareto.py. Keeps the
# visualisation stable when outlier CPT runs blow up a dataset's scale.
PARETO_XLIM = {
    150: {"starcoder": (0, 4.25), "musicpile": (0, 4.0), "tulu": (0, 3.5),
          "gsm8k": (0, 3.7),      "siqa": (0, 5.0),     "stackmathqa": (0, 4.5)},
    60:  {"starcoder": (0, 4.5),  "musicpile": (0, 4.5), "tulu": (0, 3.8),
          "gsm8k": (0, 3.8),      "siqa": (0, 4.2),     "stackmathqa": (0, 4.5)},
    20:  {"starcoder": (0, 5.5),  "musicpile": (0, 5.5), "tulu": (0, 4.2),
          "gsm8k": (0, 5.0),      "siqa": (0, 5.0),     "stackmathqa": (0, 5.5)},
}
PARETO_YLIM = {
    150: {"starcoder": (0, 2.5),  "musicpile": (0, 1.55), "tulu": (0, 2.6),
          "gsm8k": (0, 10),       "siqa": (0, 1.6),      "stackmathqa": (0, 10)},
    60:  {"starcoder": (0, 2.8),  "musicpile": (0, 1.7),  "tulu": (0, 2.8),
          "gsm8k": (0, 1.5),      "siqa": (0, 1.6),      "stackmathqa": (0, 10)},
    20:  {"starcoder": (0, 3.1),  "musicpile": (0, 2.0),  "tulu": (0, 3.5),
          "gsm8k": (0, 2.0),      "siqa": (0, 10),       "stackmathqa": (0, 1.8)},
}


def pareto_window(size: int, cpt_dataset: str):
    """Return (xlim, ylim) tuples for the (size, cpt_dataset) scatter window."""
    return PARETO_XLIM[size][cpt_dataset], PARETO_YLIM[size][cpt_dataset]


# --- loaders ---------------------------------------------------------------

def load_small_scale() -> List[dict]:
    return json.loads((DATA / "final_results_small_scale.json").read_text())


def load_midtrain_1b() -> List[dict]:
    return json.loads((DATA / "midtrain_olmo_results.json").read_text())


def load_analysis() -> List[dict]:
    return json.loads((DATA / "final_results_analysis.json").read_text())


# --- float-equality helper -------------------------------------------------

def _rho_eq(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) < 1e-9


def _sort_num(v) -> float:
    return float(v) if v is not None else float("-inf")


# --- small-scale filter helpers -------------------------------------------

def _match_small(
    r: dict, *, size, token, optimizer, pretrain_lrs="cosine",
    pretrain_lr=None, anneal_percent=None, anneal_optim=None,
    anneal_match=None, rho=None, model_type=None,
) -> bool:
    if r.get("size") != size or r.get("token") != token:
        return False
    if r.get("optimizer") != optimizer:
        return False
    if pretrain_lrs is not None and r.get("pretrain_lrs") != pretrain_lrs:
        return False
    if pretrain_lr is not None and r.get("pretrain_lr") != pretrain_lr:
        return False
    if anneal_percent is not None and r.get("anneal_percent") != anneal_percent:
        return False
    if anneal_optim is not None and r.get("anneal_optim") != anneal_optim:
        return False
    if anneal_match is not None:
        # `anneal_match == "both"` is a superset of token/compute
        if r.get("anneal_match") not in (anneal_match, "both"):
            return False
    if rho is not None and not _rho_eq(r.get("rho"), rho):
        return False
    if model_type is not None and r.get("model_type") != model_type:
        return False
    return True


def pretrain_loss(records: Iterable[dict], **filters) -> Optional[float]:
    """Return `dclm_val` of the single matching pretrain record, or None.

    If multiple records match, returns the first. Prefers `model_type=="olmo"`
    when ambiguous (matches the preference in `helper.get_run_info`).
    """
    candidates = [
        r for r in records
        if r.get("run_type") == "pretrain" and _match_small(r, **filters)
    ]
    if not candidates:
        return None
    olmo = [r for r in candidates if r.get("model_type") == "olmo"]
    pick = olmo[0] if olmo else candidates[0]
    return pick.get("dclm_val")


def pretrain_runs(records: Iterable[dict], **filters) -> List[dict]:
    """Return matching pretrain records, preferring `model_type=="olmo"` first."""
    candidates = [
        r for r in records
        if r.get("run_type") == "pretrain" and _match_small(r, **filters)
    ]
    return sorted(
        candidates,
        key=lambda r: (r.get("model_type") != "olmo", _sort_num(r.get("pretrain_lr"))),
    )


def cpt_runs(
    records: Iterable[dict], *, run_type: str = "cpt", cpt_dataset: str,
    cpt_tokens: int, cpt_lr=None, cpt_wd=None, cpt_bs=None, theta=None,
    **filters,
) -> List[dict]:
    """Return matching CPT-style records as raw dicts.

    Supports both `run_type == "cpt"` and analysis-only `run_type ==
    "interpolation"` records, which share the same fine-tuning metadata.
    """
    out = []
    for r in records:
        if r.get("run_type") != run_type:
            continue
        if r.get("cpt_dataset") != cpt_dataset or r.get("cpt_tokens") != cpt_tokens:
            continue
        if cpt_lr is not None and r.get("cpt_lr") != cpt_lr:
            continue
        if cpt_wd is not None and r.get("cpt_wd") != cpt_wd:
            continue
        if cpt_bs is not None and r.get("cpt_bs") != cpt_bs:
            continue
        if theta is not None and r.get("theta") != theta:
            continue
        if not _match_small(r, **filters):
            continue
        out.append(r)
    return sorted(
        out,
        key=lambda r: (
            _sort_num(r.get("token")),
            _sort_num(r.get("pretrain_lr")),
            _sort_num(r.get("cpt_lr")),
            _sort_num(r.get("theta")),
        ),
    )


def cpt_points(
    records: Iterable[dict], *, cpt_dataset: str, cpt_tokens: int,
    cpt_wd=None, cpt_bs=None, **filters,
) -> List[Tuple[float, float, float]]:
    """Return [(cpt_lr, dclm_val, {cpt_dataset}_val)] for matching CPT runs."""
    key = f"{cpt_dataset}_val"
    out = []
    for r in records:
        if r.get("run_type") != "cpt":
            continue
        if r.get("cpt_dataset") != cpt_dataset or r.get("cpt_tokens") != cpt_tokens:
            continue
        if cpt_wd is not None and r.get("cpt_wd") != cpt_wd:
            continue
        if cpt_bs is not None and r.get("cpt_bs") != cpt_bs:
            continue
        if not _match_small(r, **filters):
            continue
        lr, x, y = r.get("cpt_lr"), r.get("dclm_val"), r.get(key)
        if None in (lr, x, y):
            continue
        out.append((float(lr), float(x), float(y)))
    return sorted(out, key=lambda p: p[0])


def perturbed_loss(records: Iterable[dict], *, perturbation: float,
                   **filters) -> Optional[float]:
    for r in records:
        if r.get("run_type") != "perturbed":
            continue
        if abs(float(r.get("perturbation", -1)) - float(perturbation)) > 1e-12:
            continue
        if _match_small(r, **filters):
            return r.get("dclm_val")
    return None


def quant_loss(records: Iterable[dict], *, quant_bit: int = 4,
               **filters) -> Optional[float]:
    filters.setdefault("model_type", "hf")
    for r in records:
        if r.get("run_type") != "quant":
            continue
        if r.get("quant_bit") != quant_bit:
            continue
        if _match_small(r, **filters):
            return r.get("dclm_val")
    return None


def ewc_points(
    records: Iterable[dict], *, cpt_dataset: str, cpt_tokens: int,
    ewc_lambda: float, ewc_bs: int = 100, **filters,
) -> List[Tuple[float, float, float]]:
    """Return [(cpt_lr, dclm_val, {ds}_val)] for EWC runs at a given λ."""
    key = f"{cpt_dataset}_val"
    out = []
    for r in records:
        if r.get("run_type") != "ewc":
            continue
        if r.get("cpt_dataset") != cpt_dataset or r.get("cpt_tokens") != cpt_tokens:
            continue
        if r.get("ewc_lambda") != ewc_lambda or r.get("ewc_bs") != ewc_bs:
            continue
        if not _match_small(r, **filters):
            continue
        lr, x, y = r.get("cpt_lr"), r.get("dclm_val"), r.get(key)
        if None in (lr, x, y):
            continue
        out.append((float(lr), float(x), float(y)))
    return sorted(out, key=lambda p: p[0])


# --- 1B (midtrain) helpers -------------------------------------------------

def midtrain_cpt_points(
    records: Iterable[dict], *, optimizer: str, cpt_dataset: str, cpt_tokens: int,
    rho: Optional[float] = None, pretrain_token: int = 4, midtrain_tokens: int = 50,
    step: int = -1, olmes_tasks=None,
) -> List[Tuple[float, Optional[float], float, Optional[float]]]:
    """Return [(cpt_lr, mean_olmes_or_None, finetuning_val_loss, rho)] for the 1B CPT runs.

    If `olmes_tasks` is given, mean OLMES is only filled when all tasks are
    present in `olmes_eval`; otherwise it is None.
    """
    out = []
    for r in records:
        if r.get("run_type") != "cpt":
            continue
        if r.get("pretrain_token") != pretrain_token:
            continue
        if r.get("midtrain_tokens") != midtrain_tokens:
            continue
        if r.get("optimizer") != optimizer:
            continue
        if optimizer == "sam" and rho is not None and not _rho_eq(r.get("rho"), rho):
            continue
        if r.get("cpt_dataset") != cpt_dataset or r.get("cpt_tokens") != cpt_tokens:
            continue
        if r.get("step", -1) != step:
            continue
        lr = r.get("cpt_lr")
        y = r.get("finetuning_val_loss")
        if lr is None or y is None:
            continue
        mean_olmes = None
        if olmes_tasks is not None:
            oe = r.get("olmes_eval") or {}
            vals = [oe.get(t) for t in olmes_tasks]
            if None not in vals:
                mean_olmes = float(sum(vals) / len(vals))
        out.append((float(lr), mean_olmes, float(y), r.get("rho")))
    return sorted(out, key=lambda p: p[0])


def midtrain_baseline_olmes(
    records: Iterable[dict], *, optimizer: str, rho: Optional[float] = None,
    pretrain_token: int = 4, midtrain_tokens: int = 50, olmes_tasks,
    eval_types=("hf", "olmo"),
) -> Optional[float]:
    """Mean OLMES over `olmes_tasks` for the matching pre-CPT downstream run."""
    for et in eval_types:
        for r in records:
            if r.get("run_type") != "downstream":
                continue
            if r.get("pretrain_token") != pretrain_token:
                continue
            if r.get("midtrain_tokens") != midtrain_tokens:
                continue
            if r.get("optimizer") != optimizer:
                continue
            if optimizer == "sam" and rho is not None and not _rho_eq(r.get("rho"), rho):
                continue
            if r.get("eval_type") != et:
                continue
            oe = r.get("olmes_eval") or {}
            vals = [oe.get(t) for t in olmes_tasks]
            if None in vals:
                continue
            return float(sum(vals) / len(vals))
    return None
