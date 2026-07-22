"""Vessel-classification linear probes across frozen encoders (exp-13 data).

Mirrors FT-sweep-A's protocol on the recovered exp-13 task: 2,257
trajectories (512x6 raw), 4 vessel classes, original train/val/test
splits. Each encoder embeds the LAST 128 posits (mean pool, frozen);
a logistic-regression linear probe (closed-form-ish, appropriate at
n=1.6k) is fit on train, selected on val (C grid), scored on test.
Baselines: majority class, kinematic summary stats (the dumb-features
floor), random-init encoder. MLflow experiment: trackfm/ft-vessel-class.

Usage: python scripts/ft_vessel_probe.py
Writes ~/data/trackfm/ft_vessel_probe.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trackfm.config import PretrainConfig, load_config  # noqa: E402
from trackfm.models.factory import build_model  # noqa: E402

D = Path("/home/paul/data/trackfm/archive/exp13_vessel_finetunes/data/processed")
ENCODERS = [
    ("large-fixed-mlp", "scaling_large_mlp_50M", "scaling-large-mlp-50M"),
    ("large-cone-mlp", "scaling_large_cone_mlp_50M", "scaling-large-cone-mlp-50M"),
    ("xlarge-fixed", "scaling_xlarge_50M", "scaling-xlarge-50M"),
    ("large-cone", "scaling_large_cone_50M", "scaling-large-cone-50M"),
    ("large-fixed", "scaling_large_50M", "scaling-large-50M"),
    ("golden-large", "golden_large", "golden-large"),
    ("exp11-116M", "recovered_exp11_116M", "recovered-exp11-116M"),
    ("exp14-100M", "recovered_exp14_100M_h800", "recovered-exp14-100M-h800"),
    ("exp14-18M", "recovered_exp14_18M_grid12", "recovered-exp14-18M-grid12"),
    ("exp10-large", "recovered_exp10_large", "recovered-exp10-large"),
    ("small-cone", "scaling_small_cone_50M", "scaling-small-cone-50M"),
    ("ctx-geotraffic", "scaling_small_cone_ctx_geotraffic_50M",
     "scaling-small-cone-ctx-geotraffic-50M"),
    ("random-large", "scaling_large_cone_mlp_50M", None),
]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow

    x_raw = np.load(D / "trajectories.npy", allow_pickle=True)
    y = np.load(D / "labels.npy", allow_pickle=True).astype(np.int64)
    splits = json.load(open(D / "splits.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # last-128 windows, normalized with the campaign constants
    base = load_config("configs/pretrain/scaling_large_cone_mlp_50M.yaml",
                       PretrainConfig)
    nm = base.normalization
    def last128(t):
        t = np.asarray(t, dtype=np.float32)[-128:]
        if len(t) < 128:                      # 570/2257 tracks are short:
            pad = np.repeat(t[:1], 128 - len(t), axis=0)   # front-pad with
            t = np.concatenate([pad, t])                   # first posit
        return t
    X = np.stack([last128(t) for t in x_raw]).astype(np.float32)
    X[..., 0] = (X[..., 0] - nm.lat_center) / nm.lat_scale
    X[..., 1] = (X[..., 1] - nm.lon_center) / nm.lon_scale
    X[..., 2] = X[..., 2] / nm.sog_scale
    X[..., 5] = X[..., 5] / nm.dt_scale
    idx = {k: np.array(v) for k, v in splits.items()}

    def probe(emb, tag):
        best, best_f1 = None, -1
        for C in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(max_iter=2000, C=C)
            clf.fit(emb[idx["train"]], y[idx["train"]])
            f1v = f1_score(y[idx["val"]], clf.predict(emb[idx["val"]]),
                           average="macro")
            if f1v > best_f1:
                best, best_f1 = clf, f1v
        pred = best.predict(emb[idx["test"]])
        return {"accuracy": float(accuracy_score(y[idx["test"]], pred)),
                "f1_macro": float(f1_score(y[idx["test"]], pred, average="macro"))}

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("trackfm/ft-vessel-class")
    out = {}

    # baselines
    maj = np.bincount(y[idx["train"]]).argmax()
    out["baseline-majority"] = {
        "accuracy": float((y[idx["test"]] == maj).mean()),
        "f1_macro": float(f1_score(y[idx["test"]],
                                   np.full(len(idx["test"]), maj),
                                   average="macro"))}
    sog = X[..., 2]
    dcog = np.abs(np.diff(np.arctan2(X[..., 3], X[..., 4]), axis=1))
    dcog = np.minimum(dcog, 2 * np.pi - dcog)
    feats = np.stack([sog.mean(1), sog.std(1), np.quantile(sog, .9, 1),
                      dcog.mean(1), dcog.std(1), X[..., 5].mean(1),
                      X[..., 5].std(1)], axis=1)
    out["baseline-kinematic-stats"] = probe(feats, "kinstats")

    for tag, cfgname, ck in ENCODERS:
        cfg = load_config(f"configs/pretrain/{cfgname}.yaml", PretrainConfig)
        model = build_model(cfg.model, cfg.normalization)
        if ck:
            sd = torch.load(f"/home/paul/data/trackfm/checkpoints/{ck}/best.pt",
                            map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(sd)
        model.to(device).eval()
        embs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(X), 512):
                e = model.encode(torch.from_numpy(X[s:s+512]).to(device))
                embs.append(e.float().mean(dim=1).cpu().numpy())
        emb = np.concatenate(embs)
        out[tag] = probe(emb, tag)
        with mlflow.start_run(run_name=f"{tag}-lp"):
            mlflow.log_metrics({f"test_{k}": v for k, v in out[tag].items()})
            mlflow.log_param("encoder", tag)
        print(tag, out[tag], flush=True)
        del model
        torch.cuda.empty_cache()

    for k in ("baseline-majority", "baseline-kinematic-stats"):
        with mlflow.start_run(run_name=k):
            mlflow.log_metrics({f"test_{m}": v for m, v in out[k].items()})
    json.dump(out, open("/home/paul/data/trackfm/ft_vessel_probe.json", "w"),
              indent=2)
    print("written", flush=True)


if __name__ == "__main__":
    main()
