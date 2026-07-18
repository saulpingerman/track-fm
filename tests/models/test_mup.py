"""muP retrofit invariants.

The load-bearing test is the COORDINATE CHECK (canonical muP validation,
Yang & Hu App. F): under muP, activation magnitudes stay approximately
width-invariant through several aggressive optimizer steps; under SP they
drift with width. Everything else pins the mechanism: group partition and
LR/wd formulas, init-variance scaling, config validation, state-dict key
equality (muP is init+optimizer only — no new params, no forward fork).
"""
from __future__ import annotations

import math

import pytest
import torch

from trackfm.config import ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.training.mup import build_optimizer, build_param_groups

SEQ, MAX_H, N_H = 32, 64, 4


def _cfg(d_model, nhead=None, mup_enabled=False, d_base=32, **over):
    nhead = nhead if nhead is not None else max(2, d_model // 16)
    return ModelConfig(
        d_model=d_model, nhead=nhead, num_layers=2,
        dim_feedforward=4 * d_model, dropout=0.0, max_seq_len=SEQ,
        grid_size=16, grid_range=0.3, num_freqs=3,
        mup={"enabled": mup_enabled, "d_base": d_base, "d_head": 16},
        **over)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return CausalAISModel(cfg, NormalizationConfig(), max_horizon=MAX_H,
                          num_horizon_samples=N_H)


def _batch(B=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    T = SEQ + MAX_H
    f = 0.1 * torch.randn(B, T, 6, generator=g)
    f[..., 5] = 0.2 + 0.5 * torch.rand(B, T, generator=g)
    return f


class _Train:
    learning_rate = 3e-4
    weight_decay = 1e-5


# ------------------------------------------------------------- mechanism

def test_optimizer_single_group_when_mup_off():
    cfg = _cfg(64)
    model = _model(cfg)
    opt = build_optimizer(model, _Train(), cfg)
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["lr"] == 3e-4 and g["weight_decay"] == 1e-5
    assert g["betas"] == (0.9, 0.999)
    ids = {id(p) for p in g["params"]}
    assert ids == {id(p) for p in model.parameters()}


@pytest.mark.parametrize("head,mlp", [("fourier", 0), ("fourier", 24),
                                       ("direct", 0), ("direct", 24)])
def test_param_group_partition_and_lrs(head, mlp):
    cfg = _cfg(64, mup_enabled=True, d_base=32, head_type=head,
               head_mlp_hidden=mlp)
    model = _model(cfg)
    groups = build_param_groups(model, 64, 32, lr=1e-3, weight_decay=1e-4)
    m = 2.0
    assert [g["lr"] for g in groups] == [1e-3, 1e-3 / m, 1e-3 / m]
    assert [g["weight_decay"] for g in groups] == [1e-4, 1e-4 * m, 1e-4 * m]
    assert len(groups[2]["params"]) == 1            # exactly one readout tensor
    # union == trainable params, no overlap (also asserted inside builder)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}


def test_scheduler_composes_with_groups():
    from trackfm.training.pretrain import _lr_lambda
    cfg = _cfg(64, mup_enabled=True, d_base=32)
    model = _model(cfg)
    opt = build_optimizer(model, _Train(), cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, 10, 100, "cosine"))
    for _ in range(5):                               # mid-warmup: factor 5/10
        sched.step()
    lrs = [g["lr"] for g in opt.param_groups]
    lam = _lr_lambda(5, 10, 100, "cosine")
    assert math.isclose(lrs[0], 3e-4 * lam, rel_tol=1e-9)
    assert math.isclose(lrs[1], 3e-4 / 2 * lam, rel_tol=1e-9)


def test_init_variance_scaling():
    torch.manual_seed(0)
    base = _model(_cfg(64, mup_enabled=True, d_base=64))
    torch.manual_seed(0)
    wide = _model(_cfg(256, mup_enabled=True, d_base=64))

    def readout_std(m):
        head = m.fourier_head.coeff_predictor
        w = head.weight if isinstance(head, torch.nn.Linear) else head[0].weight
        return w.std().item()

    # output: std ratio EXACTLY d_base/d (1/fan_in^2 variance rule)
    r = readout_std(wide) / readout_std(base)
    assert abs(r - 64 / 256) < 0.05 * (64 / 256 + 1), r
    # muP@d_base == SP@d_base: readout std at base width is the SP 0.01
    assert abs(readout_std(base) - 0.01) < 0.001

    # hidden: default kaiming gives std ~ 1/sqrt(fan_in) already
    h_b = base.horizon_proj.weight.std().item()
    h_w = wide.horizon_proj.weight.std().item()
    assert abs(h_w / h_b - math.sqrt(64 / 256)) < 0.1

    # input: width-invariant std
    i_b = base.input_proj.weight.std().item()
    i_w = wide.input_proj.weight.std().item()
    assert abs(i_w / i_b - 1.0) < 0.15


def test_readout_zero_init_variant():
    cfg = _cfg(64, mup_enabled=True, d_base=32)
    cfg.mup.readout_zero_init = True
    model = _model(cfg)
    head = model.fourier_head.coeff_predictor
    w = head.weight if isinstance(head, torch.nn.Linear) else head[0].weight
    assert int(w.count_nonzero()) == 0


def test_config_validator_refusals():
    with pytest.raises(ValueError, match="constant d_head"):
        ModelConfig(d_model=768, nhead=16, dim_feedforward=3072,
                    mup={"enabled": True})
    with pytest.raises(ValueError, match="dim_feedforward"):
        ModelConfig(d_model=768, nhead=48, dim_feedforward=2048,
                    mup={"enabled": True})
    # same shapes load fine when disabled
    ModelConfig(d_model=768, nhead=16, dim_feedforward=3072,
                mup={"enabled": False})


def test_state_dict_keys_identical_on_off():
    """muP is init+optimizer only: no new params, no forward fork."""
    off = _model(_cfg(64, mup_enabled=False))
    on = _model(_cfg(64, mup_enabled=True, d_base=32))
    assert set(off.state_dict().keys()) == set(on.state_dict().keys())
    # and a checkpoint round-trip across the flag strict-loads
    on.load_state_dict(off.state_dict())


# ------------------------------------------------------- coordinate check

def _act_scales(model, batch, n_steps=3, lr=1e-2):
    """Mean |activation| at input_proj out / encoder out / head logits,
    measured after n aggressive AdamW steps."""
    from trackfm.training.losses import compute_soft_target_loss

    class _T:
        learning_rate = lr
        weight_decay = 0.0

    cfg_holder = model.model_cfg
    opt = build_optimizer(model, _T(), cfg_holder)
    model.train()
    for _ in range(n_steps):
        ld, tgt, _, _ = model.forward_train(batch, causal=True)
        loss = compute_soft_target_loss(ld.float(), tgt.float(),
                                        cfg_holder.grid_range,
                                        cfg_holder.grid_size, 0.003)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    acts = {}
    with torch.no_grad():
        x = batch[:, :SEQ, :]
        h = model.input_proj(x)
        acts["input_proj"] = h.abs().mean().item()
        enc = model.encode(x)
        acts["encoder_out"] = enc.abs().mean().item()
        ld, _, _, _ = model.forward_train(batch,
                                          horizon_indices=torch.tensor([4, 16]),
                                          causal=False)
        acts["log_density_range"] = (ld.max() - ld.min()).item()
    return acts


def _slope(widths, values):
    xs = [math.log(w) for w in widths]
    ys = [math.log(max(v, 1e-12)) for v in values]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def test_coord_check_width_invariance():
    """Activations stay ~width-invariant under muP after aggressive steps;
    SP must show real drift somewhere or this test has no teeth."""
    widths = [32, 64, 128]
    batch = _batch()

    mup_acts, sp_acts = {}, {}
    for w in widths:
        torch.manual_seed(0)
        mup_acts[w] = _act_scales(
            _model(_cfg(w, mup_enabled=True, d_base=32)), batch)
        torch.manual_seed(0)
        sp_acts[w] = _act_scales(
            _model(_cfg(w, mup_enabled=False)), batch)

    for key in ("input_proj", "encoder_out", "log_density_range"):
        s_mup = abs(_slope(widths, [mup_acts[w][key] for w in widths]))
        assert s_mup <= 0.35, (
            f"muP activation drift at {key}: |slope|={s_mup:.3f} "
            f"(values {[mup_acts[w][key] for w in widths]})")

    # power check: SP must drift somewhere (the readout temperature bug
    # guarantees log-density range grows with width under SP)
    sp_slopes = {k: abs(_slope(widths, [sp_acts[w][k] for w in widths]))
                 for k in ("input_proj", "encoder_out", "log_density_range")}
    assert max(sp_slopes.values()) >= 0.30, (
        f"SP shows no width drift anywhere ({sp_slopes}) — "
        f"coordinate check has no discriminating power")


def test_wd_wiring_through_build_optimizer():
    """Mutation testing (verify-must-fix 4) showed hardcoding
    independent_wd=False inside build_optimizer passes every other test —
    the wd wiring through the PRODUCTION entry point was never asserted.
    A regression would train the width series with effective decay ~m x
    too weak on B/C at d=768 with green CI."""
    cfg = _cfg(64, mup_enabled=True, d_base=32)
    opt = build_optimizer(_model(cfg), _Train(), cfg)
    m = 2.0
    assert [g["weight_decay"] for g in opt.param_groups] == \
        [1e-5, 1e-5 * m, 1e-5 * m]

    cfg2 = _cfg(64, mup_enabled=True, d_base=32)
    cfg2.mup.independent_wd = False
    opt2 = build_optimizer(_model(cfg2), _Train(), cfg2)
    assert [g["weight_decay"] for g in opt2.param_groups] == \
        [1e-5, 1e-5, 1e-5]


def test_coord_check_step0_readout_temperature():
    """Step-0 probe (verify hardening): the coordinate check at step 3 has
    no power against the readout-init mutation (eta/m LR washes the
    temperature out within steps). At step 0 the separation is clean:
    log-density range must NOT grow with width under muP."""
    widths = [32, 64, 128]
    ranges = []
    for w in widths:
        torch.manual_seed(0)
        model = _model(_cfg(w, mup_enabled=True, d_base=32))
        model.eval()
        with torch.no_grad():
            ld, _, _, _ = model.forward_train(
                _batch(), horizon_indices=torch.tensor([4, 16]), causal=False)
        ranges.append((ld.max() - ld.min()).item())
    s = _slope(widths, ranges)
    assert s <= 0.35, f"readout temperature grows with width: slope={s:.3f} " \
                      f"ranges={ranges} (init variance rule broken?)"
