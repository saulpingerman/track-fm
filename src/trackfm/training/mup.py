"""muP optimizer construction (Yang & Hu 2022, Tensor Programs V).

Three parameter groups under muP, one width multiplier m = d_model/d_base:

  A (input-like / width-invariant): lr = eta,     wd = lambda
     - every bias and LayerNorm vector (ndim <= 1)
     - input_proj.weight, time_proj.0.weight (fixed fan_in: 6, 1)
     - context_bias.* (width-independent CNN)
     - head-MLP second linear (neither dim touches d_model)
  B (hidden matrices, fan_in ~ d_model): lr = eta/m, wd = lambda*m
     - transformer QKV/out/FFN weights, time_proj.2, horizon_proj
  C (output readout, fan_in = d_model, fan_out fixed): lr = eta/m, wd = lambda*m
     - fourier_head coeff_predictor / DirectGridHead logits (d_in-facing linear)

AdamW column of Yang & Hu Table 3/8: Adam's normalized updates make
per-entry deltas Theta(lr) regardless of gradient scale, and the
correlated sum over ~d coordinates forces lr ~ 1/d (LINEAR in width;
1/sqrt appears in NO muP column — it is the no-alignment heuristic).

wd*m on down-LR'd groups cancels PyTorch AdamW's lr-COUPLED decay
(p -= lr*wd*p), keeping effective decay width-invariant
(mup.independent_wd, default True).

The classifier is EXHAUSTIVE and raises on unknown parameters: adding a
module without classifying it breaks CI, not the flagship.

Downstream fine-tune wrappers (heads.py) are classified too: the
'encoder.' prefix is stripped so backbone params reuse the rules above,
and the fresh MLPHead gets explicit roles (see _DOWNSTREAM_*).
build_finetune_optimizer is the finetune trainer's entry point.

When mup.enabled is False this module returns the verbatim single-group
AdamW the trainer has always built — structurally identical, not merely
equivalent (SP bit-for-bit gate: tests/training/test_sp_equivalence.py).
"""
from __future__ import annotations

import re

import torch

_HIDDEN_EXACT = {"time_proj.2.weight", "horizon_proj.weight"}
_HIDDEN_RE = re.compile(
    r"^transformer\.layers\.\d+\."
    r"(self_attn\.in_proj_weight|self_attn\.out_proj\.weight"
    r"|linear1\.weight|linear2\.weight)$")
_INPUT_EXACT = {"input_proj.weight", "time_proj.0.weight"}
# head readout: the d_in-facing linear for BOTH head types, both
# projector shapes (plain Linear, or Sequential index 0 when
# head_mlp_hidden > 0). Attribute is `fourier_head` for both types.
_OUTPUT_EXACT = {
    "fourier_head.coeff_predictor.weight",
    "fourier_head.coeff_predictor.0.weight",
    "fourier_head.logits.weight",
    "fourier_head.logits.0.weight",
}
# head-MLP layer 2: (out_dim, head_mlp_hidden) — neither dim scales with
# d_model. A naive "last layer = output" rule would misclassify this.
_HEAD_MLP2_EXACT = {
    "fourier_head.coeff_predictor.2.weight",
    "fourier_head.logits.2.weight",
}

# Downstream wrapper models (PortClassifier / EtaRegressor) prefix the
# backbone with "encoder." and add an MLPHead:
#   head.net.0 Linear(d, d/2)    — both dims scale with width -> hidden
#   head.net.3 Linear(d/2, 128)  — fan_in scales, fan_out fixed -> output
#   head.net.6 Linear(128, out)  — width-independent -> group A
# (biases are ndim<=1 -> group A automatically.)
_WRAPPER_PREFIX = "encoder."
_DOWNSTREAM_HIDDEN_EXACT = {"head.net.0.weight"}
_DOWNSTREAM_OUTPUT_EXACT = {"head.net.3.weight"}
_DOWNSTREAM_A_EXACT = {"head.net.6.weight"}


def build_param_groups(model: torch.nn.Module, d_model: int, d_base: int,
                       lr: float, weight_decay: float,
                       independent_wd: bool = True) -> list[dict]:
    """Split named_parameters into muP groups A/B/C. Raises on unknowns."""
    m = d_model / d_base
    group_a, group_b, group_c = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # downstream wrappers prefix the backbone; strip it so the
        # pretraining rules below apply verbatim to encoder.* params
        base = name.removeprefix(_WRAPPER_PREFIX)
        if p.ndim <= 1:
            group_a.append(p)                       # biases, LayerNorm vectors
        elif base in _INPUT_EXACT or base.startswith("context_bias."):
            group_a.append(p)
        elif base in _HIDDEN_EXACT or _HIDDEN_RE.match(base):
            group_b.append(p)
        elif base in _OUTPUT_EXACT:
            group_c.append(p)
        elif base in _HEAD_MLP2_EXACT:
            group_a.append(p)
        elif name in _DOWNSTREAM_HIDDEN_EXACT:
            group_b.append(p)
        elif name in _DOWNSTREAM_OUTPUT_EXACT:
            group_c.append(p)
        elif name in _DOWNSTREAM_A_EXACT:
            group_a.append(p)
        else:
            hint = (" (torch.compile wraps names with _orig_mod. — build "
                    "the optimizer BEFORE compiling the model)"
                    if name.startswith("_orig_mod.") else "")
            raise ValueError(
                f"muP: unclassified parameter {name!r} — every new module "
                f"must be deliberately assigned a muP role in "
                f"trackfm/training/mup.py before it can train under muP."
                f"{hint}")

    wd_scaled = weight_decay * m if independent_wd else weight_decay
    groups = [
        {"params": group_a, "lr": lr, "weight_decay": weight_decay},
        {"params": group_b, "lr": lr / m, "weight_decay": wd_scaled},
        {"params": group_c, "lr": lr / m, "weight_decay": wd_scaled},
    ]

    # completeness: union of groups == trainable params, no duplicates
    grouped = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(set(grouped)), "muP: parameter in two groups"
    assert set(grouped) == set(trainable), "muP: group union != model params"
    return groups


def build_optimizer(model: torch.nn.Module, train_cfg,
                    model_cfg) -> torch.optim.AdamW:
    """The trainer's single entry point for optimizer construction.

    mup disabled -> the VERBATIM historical call (same iterator, one
    group); enabled -> three muP groups per build_param_groups.
    """
    if not model_cfg.mup.enabled:
        return torch.optim.AdamW(model.parameters(),
                                 lr=train_cfg.learning_rate,
                                 weight_decay=train_cfg.weight_decay)
    groups = build_param_groups(
        model, d_model=model_cfg.d_model, d_base=model_cfg.mup.d_base,
        lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
        independent_wd=model_cfg.mup.independent_wd)
    return torch.optim.AdamW(groups)


def build_finetune_optimizer(model: torch.nn.Module, train_cfg, model_cfg,
                             lr: float | None = None) -> torch.optim.AdamW:
    """Optimizer entry point for downstream fine-tune models (heads.py
    wrappers: backbone under 'encoder.', fresh MLPHead under 'head.').

    mup disabled -> the VERBATIM historical finetune call (requires_grad
    filter, single group); enabled -> the same three muP groups.
    build_param_groups already skips frozen params, so the LP phase
    (frozen backbone) yields head-only groups with no special casing.

    `lr` overrides train_cfg.learning_rate: LP-FT phases run at
    different LRs off one config.
    """
    lr = train_cfg.learning_rate if lr is None else lr
    if not model_cfg.mup.enabled:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=train_cfg.weight_decay)
    groups = build_param_groups(
        model, d_model=model_cfg.d_model, d_base=model_cfg.mup.d_base,
        lr=lr, weight_decay=train_cfg.weight_decay,
        independent_wd=model_cfg.mup.independent_wd)
    return torch.optim.AdamW(groups)
