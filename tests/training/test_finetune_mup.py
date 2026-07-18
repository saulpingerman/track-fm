"""Downstream muP wiring + LP-FT scheduling.

The finetune wrappers (PortClassifier/EtaRegressor) prefix the backbone
with 'encoder.' and add a fresh MLPHead; build_finetune_optimizer must
classify BOTH under muP, and must reproduce the verbatim historical
single-group call (requires_grad filter included) when muP is off — the
finetune analogue of the SP bit-preservation gate.

LP-FT (Kumar et al. 2022): probe phase trains the head to early-stop with
the backbone frozen, the BEST probe head is reloaded, the backbone
unfreezes, phase 2 runs at the lower FT LR. The loop test spies on
optimizer construction to pin exactly that sequence.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from trackfm.config import MLflowConfig, ModelConfig, NormalizationConfig
from trackfm.models.encoder import CausalAISModel
from trackfm.models.heads import EtaRegressor, PortClassifier
from trackfm.training.mup import build_finetune_optimizer

SEQ = 16


def _cfg(d_model=32, mup_enabled=True, d_base=32):
    return ModelConfig(
        d_model=d_model, nhead=max(2, d_model // 16), num_layers=1,
        dim_feedforward=4 * d_model, dropout=0.0, max_seq_len=SEQ,
        grid_size=8, grid_range=0.3, num_freqs=2,
        mup={"enabled": mup_enabled, "d_base": d_base, "d_head": 16})


def _classifier(cfg, num_classes=3, freeze=False, seed=0):
    torch.manual_seed(seed)
    enc = CausalAISModel(cfg, NormalizationConfig(), max_horizon=8,
                         num_horizon_samples=2)
    return PortClassifier(enc, num_classes, "mean", 0.3, freeze)


class _Train:
    learning_rate = 2e-4
    weight_decay = 1e-5


def _name_to_group(model, opt):
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            out[id_to_name[id(p)]] = gi
    return out


# ------------------------------------------------------------ partition

def test_off_path_single_group_filters_frozen():
    """muP off -> the VERBATIM historical call: one group, ONLY
    requires_grad params (the old code filtered; build_optimizer's
    pretrain off-path does not — the distinction is load-bearing here)."""
    cfg = _cfg(mup_enabled=False)
    model = _classifier(cfg, freeze=True)
    opt = build_finetune_optimizer(model, _Train(), cfg)
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    assert g["lr"] == 2e-4 and g["weight_decay"] == 1e-5
    assert g["betas"] == (0.9, 0.999)
    ids = {id(p) for p in g["params"]}
    assert ids == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(n.startswith("head.") for n, p in model.named_parameters()
               if id(p) in ids)


@pytest.mark.parametrize("wrapper", ["port", "eta"])
def test_downstream_partition_full(wrapper):
    cfg = _cfg(d_model=64, d_base=32)
    torch.manual_seed(0)
    enc = CausalAISModel(cfg, NormalizationConfig(), max_horizon=8,
                         num_horizon_samples=2)
    model = (PortClassifier(enc, 3) if wrapper == "port"
             else EtaRegressor(enc))
    opt = build_finetune_optimizer(model, _Train(), cfg)
    m = 2.0
    assert [g["lr"] for g in opt.param_groups] == [2e-4, 2e-4 / m, 2e-4 / m]
    assert [g["weight_decay"] for g in opt.param_groups] == \
        [1e-5, 1e-5 * m, 1e-5 * m]

    grp = _name_to_group(model, opt)
    # fresh MLPHead roles
    assert grp["head.net.0.weight"] == 1          # (d, d/2): hidden
    assert grp["head.net.3.weight"] == 2          # (d/2, 128): output
    assert grp["head.net.6.weight"] == 0          # (128, out): width-free
    assert grp["head.net.0.bias"] == 0
    # backbone rules survive the encoder. prefix
    assert grp["encoder.input_proj.weight"] == 0
    assert grp["encoder.transformer.layers.0.self_attn.in_proj_weight"] == 1
    assert grp["encoder.horizon_proj.weight"] == 1
    assert grp["encoder.fourier_head.coeff_predictor.weight"] == 2
    # completeness
    assert set(grp) == {n for n, p in model.named_parameters()
                        if p.requires_grad}


def test_downstream_partition_frozen_lp():
    """LP phase: frozen backbone -> head-only groups, same role split."""
    cfg = _cfg(d_model=64, d_base=32)
    model = _classifier(cfg, freeze=True)
    opt = build_finetune_optimizer(model, _Train(), cfg)
    grp = _name_to_group(model, opt)
    assert set(grp) == {n for n, p in model.named_parameters()
                        if p.requires_grad}
    assert all(n.startswith("head.") for n in grp)
    assert grp["head.net.0.weight"] == 1
    assert grp["head.net.3.weight"] == 2
    assert grp["head.net.6.weight"] == 0


def test_lr_override():
    cfg = _cfg(d_model=64, d_base=32)
    opt = build_finetune_optimizer(_classifier(cfg), _Train(), cfg, lr=5e-4)
    assert [g["lr"] for g in opt.param_groups] == [5e-4, 2.5e-4, 2.5e-4]
    cfg_off = _cfg(mup_enabled=False)
    opt2 = build_finetune_optimizer(_classifier(cfg_off), _Train(), cfg_off,
                                    lr=5e-4)
    assert opt2.param_groups[0]["lr"] == 5e-4


def test_unclassified_downstream_param_raises():
    cfg = _cfg()
    model = _classifier(cfg)
    model.rogue = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="unclassified parameter 'rogue"):
        build_finetune_optimizer(model, _Train(), cfg)


def test_lp_ft_freeze_encoder_conflict_rejected():
    from trackfm.training.finetune import FinetuneTrainConfig
    with pytest.raises(ValueError, match="freeze_encoder"):
        FinetuneTrainConfig(strategy="lp-ft", freeze_encoder=True)
    FinetuneTrainConfig(strategy="lp", freeze_encoder=True)   # fine


# -------------------------------------------------------------- LP-FT loop

def _patch_mlflow(ft, monkeypatch, steps=None):
    monkeypatch.setattr(ft, "start_run", lambda *a, **k: SimpleNamespace(
        info=SimpleNamespace(run_id="test")))

    def log_metrics(metrics, step=None):
        if steps is not None:
            steps.append(step)
    monkeypatch.setattr(ft.mlflow, "log_metrics", log_metrics)
    monkeypatch.setattr(ft.mlflow, "end_run", lambda *a, **k: None)


def _spy_optimizer(ft, monkeypatch):
    calls = []
    real = ft.build_finetune_optimizer

    def spy(model, train_cfg, model_cfg, lr=None):
        opt = real(model, train_cfg, model_cfg, lr=lr)
        n = sum(p.numel() for g in opt.param_groups for p in g["params"])
        calls.append({"lr": lr, "n_params": n, "model": model})
        return opt
    monkeypatch.setattr(ft, "build_finetune_optimizer", spy)
    return calls


def _tiny_datasets(ft):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(48, SEQ, 6)).astype(np.float32) * 0.1
    y = rng.integers(0, 3, size=48)
    return {k: ft.ArrayTaskDataset(x[a:b], y[a:b])
            for k, (a, b) in {"train": (0, 32), "val": (32, 40),
                              "test": (40, 48)}.items()}


def _head_n():
    return sum(p.numel() for n, p in _classifier(_cfg()).named_parameters()
               if n.startswith("head."))


def _finetune_cfg(ft, tmp_path, run_name, **train_over):
    return ft.FinetuneConfig(
        task="classification", label_column="y", data_dir=tmp_path,
        checkpoint_dir=tmp_path, model=_cfg(),
        train=ft.FinetuneTrainConfig(
            batch_size=16, num_workers=0, precision="fp32",
            warmup_epochs=1, **train_over),
        mlflow=MLflowConfig(experiment="test", run_name=run_name))


def test_lp_ft_two_phase_loop(tmp_path, monkeypatch):
    """End-to-end on synthetic data: probe phase sees ONLY head params,
    FT phase sees the full model at learning_rate/10 (default), backbone
    ends unfrozen, best.pt exists and test metrics come back."""
    import trackfm.training.finetune as ft

    _patch_mlflow(ft, monkeypatch)
    calls = _spy_optimizer(ft, monkeypatch)

    loads = []
    real_load = torch.load

    def load_spy(path, *a, **k):
        loads.append(str(path))
        return real_load(path, *a, **k)
    monkeypatch.setattr(torch, "load", load_spy)

    cfg = _finetune_cfg(ft, tmp_path, "lp_ft_test", strategy="lp-ft",
                        lp_max_epochs=2, lp_patience=2, max_epochs=2,
                        early_stopping_patience=2)
    torch.manual_seed(0)
    metrics = ft.run_finetune(cfg, _tiny_datasets(ft), num_classes=3)

    assert len(calls) == 2, "expected one optimizer per phase"
    assert calls[0]["lr"] == pytest.approx(2e-4)    # probe at learning_rate
    assert calls[0]["n_params"] == _head_n()        # probe: head only
    assert calls[1]["n_params"] > _head_n()         # FT: full model
    assert calls[1]["lr"] == pytest.approx(2e-4 / 10)
    # backbone must END unfrozen — the whole point of phase 2
    assert all(p.requires_grad
               for p in calls[-1]["model"].encoder.parameters())
    assert "f1_macro" in metrics
    assert (tmp_path / "lp_ft_test" / "best.pt").exists()
    # best.pt must be loaded TWICE: at the phase boundary (converged probe
    # head is the LP-FT protection — skipping it degrades to plain
    # staged unfreezing) and in the final best-model test reload.
    assert sum("best.pt" in p for p in loads) == 2


def test_lp_strategy_freezes_backbone(tmp_path, monkeypatch):
    """strategy='lp' THROUGH run_finetune: one phase, backbone frozen the
    whole run, optimizer sees only head params at learning_rate.
    (Verification mutant: dropping 'lp' from the frozen-strategy
    derivation shipped 'probe' results that were silent full FT.)"""
    import trackfm.training.finetune as ft

    _patch_mlflow(ft, monkeypatch)
    calls = _spy_optimizer(ft, monkeypatch)

    cfg = _finetune_cfg(ft, tmp_path, "lp_test", strategy="lp",
                        max_epochs=1, early_stopping_patience=1)
    torch.manual_seed(0)
    ft.run_finetune(cfg, _tiny_datasets(ft), num_classes=3)

    assert len(calls) == 1
    assert calls[0]["n_params"] == _head_n()
    assert calls[0]["lr"] == pytest.approx(2e-4)    # lp_* fields NOT used
    assert not any(p.requires_grad
                   for p in calls[0]["model"].encoder.parameters())


def test_lp_ft_checkpoint_and_step_contracts(tmp_path, monkeypatch):
    """Scripted val scores pin what the smoke test cannot:
    (1) best.pt written ONCE — an FT phase that never beats the converged
        probe must not overwrite it (overall-val-winner shipping);
    (2) FT still gets its full phase-local patience budget below the
        probe's score (early stop must not race the global best);
    (3) mlflow steps strictly sequential across an early-stopped probe
        phase (no cross-phase step collision);
    (4) probe runs on lp_max_epochs/lp_patience/lp_learning_rate, FT on
        max_epochs/early_stopping_patience/ft_learning_rate — budgets
        differ here, so aliasing the config pairs fails this test."""
    import trackfm.training.finetune as ft

    steps = []
    _patch_mlflow(ft, monkeypatch, steps=steps)
    calls = _spy_optimizer(ft, monkeypatch)

    scores = iter([0.9, 0.8, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1])
    monkeypatch.setattr(
        ft, "classification_metrics",
        lambda logits, y: (lambda s: {"accuracy": s, "f1_macro": s})(
            next(scores)))

    saves = []
    real_save = torch.save

    def save_spy(obj, path, *a, **k):
        saves.append(str(path))
        return real_save(obj, path, *a, **k)
    monkeypatch.setattr(torch, "save", save_spy)

    cfg = _finetune_cfg(ft, tmp_path, "lp_ft_scripted", strategy="lp-ft",
                        lp_max_epochs=3, lp_patience=1,
                        lp_learning_rate=3e-4, ft_learning_rate=5e-5,
                        max_epochs=4, early_stopping_patience=2)
    torch.manual_seed(0)
    ft.run_finetune(cfg, _tiny_datasets(ft), num_classes=3)

    # LP: 0.9* then 0.8 -> lp_patience=1 break after 2 epochs.
    # FT: 0.5 (phase-best, below probe) / 0.45 / 0.4 -> patience-2 break
    # after 3 epochs. Then the stepless test-split log (0.3).
    assert steps == [0, 1, 2, 3, 4, None]
    assert sum("best.pt" in p for p in saves) == 1  # probe stays the winner
    assert [c["lr"] for c in calls] == [pytest.approx(3e-4),
                                        pytest.approx(5e-5)]
