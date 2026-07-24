"""Fair baselines for port-dest + ETA: window-summary stats only.

User rule: a baseline is fair only if it consumes exactly what the model
consumes — the 128-posit window. No origin labels, no vessel identity.
(dest-given-origin lookup and origin-copy are demoted to privileged-
reference footnotes.)

Features (16, all from the window): first/last lat/lon, displacement,
mean/std/last sog, last cog sin/cos, mean |dCOG|, mean/std dt, path
length, straightness. Heads mirror the ladder: LINEAR and 2-layer MLP
(d->d/2->128 equivalent scaled for 16-dim input: 16->64->32->out).

Splits/caps match the FT protocol (train 300k seed 17). CPU-only,
RSS < 10 GB (safe alongside a running FT under the RAM budget).

Usage: python scripts/fair_window_baselines.py
Writes ~/data/trackfm/fair_window_baselines.json + MLflow runs
(trackfm/ft-port-dest, trackfm/ft-eta: baseline-window-stats-{linear,mlp}).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

D = Path("/home/paul/data/trackfm/ports/v2")
SEED, TRAIN_CAP = 17, 300_000


def load_split(split: str, label: str, cap: int | None):
    # port "features" = Array(f32, 640) = 128 x 5 RAW [lat,lon,sog,cog_deg,dt]
    # (same as WindowTaskDataset: reshape to NUM_FEATURES=5 then
    # normalize_features -> 128 x 6 [lat,lon,sog,cog_sin,cog_cos,dt]).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from trackfm.datasets.windowing import normalize_features
    shards = sorted((D / "port_windows").glob("shard_*.parquet"))
    df = (pl.scan_parquet(shards).filter(pl.col("split") == split)
          .select(["features", label]).collect())
    if cap and df.height > cap:
        df = df.sample(cap, seed=SEED)
    raw = df["features"].to_numpy().reshape(-1, 128, 5).astype(np.float32)
    x = normalize_features(raw)          # -> (N, 128, 6), cog->sin/cos
    return x, df[label]


def window_stats(x: np.ndarray) -> np.ndarray:
    lat, lon, sog = x[..., 0], x[..., 1], x[..., 2]
    cs, cc, dt = x[..., 3], x[..., 4], x[..., 5]
    dcog = np.abs(np.diff(np.arctan2(cs, cc), axis=1))
    dcog = np.minimum(dcog, 2 * np.pi - dcog)
    steps = np.sqrt(np.diff(lat, axis=1) ** 2 + np.diff(lon, axis=1) ** 2)
    path = steps.sum(1)
    disp = np.sqrt((lat[:, -1] - lat[:, 0]) ** 2 + (lon[:, -1] - lon[:, 0]) ** 2)
    f = np.stack([
        lat[:, 0], lon[:, 0], lat[:, -1], lon[:, -1],
        lat[:, -1] - lat[:, 0], lon[:, -1] - lon[:, 0],
        sog.mean(1), sog.std(1), sog[:, -1],
        cs[:, -1], cc[:, -1], dcog.mean(1),
        dt.mean(1), dt.std(1), path,
        disp / np.maximum(path, 1e-6),
    ], axis=1).astype(np.float32)
    return np.nan_to_num(f)


def make_head(d_in: int, d_out: int, mlp: bool) -> torch.nn.Module:
    if not mlp:
        return torch.nn.Linear(d_in, d_out)
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, d_out))


def fit(task, feats, ys, n_out, mlp):
    from sklearn.metrics import f1_score
    torch.manual_seed(SEED)
    mu, sd = feats["train"].mean(0, keepdims=True), feats["train"].std(0, keepdims=True) + 1e-6
    t = {s: torch.from_numpy((feats[s] - mu) / sd) for s in feats}
    model = make_head(t["train"].shape[1], n_out, mlp)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ytr = torch.from_numpy(ys["train"])
    best, best_val, patience = None, None, 0
    for epoch in range(30):
        for i in torch.randperm(len(ytr)).split(8192):
            if task == "cls":
                loss = torch.nn.functional.cross_entropy(model(t["train"][i]), ytr[i])
            else:
                loss = torch.nn.functional.huber_loss(
                    model(t["train"][i]).squeeze(-1), torch.log1p(ytr[i]))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pv = model(t["val"])
        if task == "cls":
            v = f1_score(ys["val"], pv.argmax(1).numpy(), average="macro")
            better = best_val is None or v > best_val
        else:
            pred_s = np.expm1(np.clip(pv.squeeze(-1).numpy(), 0, 20))
            v = float(np.abs(pred_s - ys["val"]).mean() / 60)
            better = best_val is None or v < best_val
        if better:
            best_val, patience = v, 0
            best = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    model.load_state_dict(best)
    with torch.no_grad():
        pt = model(t["test"])
    if task == "cls":
        from sklearn.metrics import accuracy_score
        pred = pt.argmax(1).numpy()
        return {"f1_macro": float(f1_score(ys["test"], pred, average="macro")),
                "accuracy": float(accuracy_score(ys["test"], pred))}
    pred_s = np.expm1(np.clip(pt.squeeze(-1).numpy(), 0, 20))
    return {"mae_minutes": float(np.abs(pred_s - ys["test"]).mean() / 60),
            "medae_minutes": float(np.median(np.abs(pred_s - ys["test"])) / 60)}


def main():
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    out = {}

    vocab = json.load(open(D / "labels.json"))
    other = vocab.get("OTHER", len(vocab) - 1)
    feats, ys_pd, ys_eta = {}, {}, {}
    for s in ("train", "val", "test"):
        x, lab = load_split(s, "destination", TRAIN_CAP if s == "train" else None)
        feats[s] = window_stats(x)
        ys_pd[s] = np.array([vocab.get(v, other) for v in lab.to_list()], dtype=np.int64)
        _, rem = load_split(s, "remaining_s", TRAIN_CAP if s == "train" else None)
        ys_eta[s] = rem.to_numpy().astype(np.float32)
        print(s, feats[s].shape, flush=True)

    for mlp in (False, True):
        tag = "mlp" if mlp else "linear"
        out[f"pd-window-stats-{tag}"] = fit("cls", feats, ys_pd, len(vocab), mlp)
        out[f"eta-window-stats-{tag}"] = fit("reg", feats, ys_eta, 1, mlp)
        print(tag, out[f"pd-window-stats-{tag}"], out[f"eta-window-stats-{tag}"], flush=True)
        for exp_name, key, metrics in (
                ("trackfm/ft-port-dest", f"pd-window-stats-{tag}", out[f"pd-window-stats-{tag}"]),
                ("trackfm/ft-eta", f"eta-window-stats-{tag}", out[f"eta-window-stats-{tag}"])):
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=f"baseline-window-stats-{tag}"):
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                mlflow.log_param("fair", "window-only features")
    json.dump(out, open("/home/paul/data/trackfm/fair_window_baselines.json", "w"),
              indent=2)
    print("written", flush=True)


if __name__ == "__main__":
    main()
