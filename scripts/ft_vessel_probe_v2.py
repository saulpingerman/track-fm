"""Vessel-classification linear probes on the vessel-v2 task (own corpus).

Replaces the exp-13 4-class probe (n too small to rank encoders) with the
vessel_v2 dataset: ~700k val-period windows, vessel-disjoint splits, AIS
ship-type labels. Classes are filtered at probe time: the junk label 1"
is dropped and only classes with >=500 train / >=100 val / >=100 test
windows survive (vessel-disjoint hashing strands rare classes' few
vessels in one split).

At this n a torch linear probe (AdamW, standardized features, val-selected
over a small LR grid) replaces sklearn LogReg. Encoders are frozen;
embeddings are mean-pooled over the 128-posit window. Baselines: majority,
kinematic summary stats, random-init encoder. MLflow experiment:
trackfm/ft-vessel-class, run names *-v2-lp.

GPU-heavy (704k encoder forwards) — run at GPU drain, not beside pretrain.

Usage: python scripts/ft_vessel_probe_v2.py
Writes ~/data/trackfm/ft_vessel_probe_v2.json.
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

D = Path("/home/paul/data/trackfm/vessel_v2")
MIN_PER_SPLIT = {"train": 500, "val": 100, "test": 100}
DROP_LABELS = {'1"'}
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
    ("cone-spectrum", "scaling_large_cone_spectrum_50M",
     "scaling-large-cone-spectrum-50M"),
    ("random-large", "scaling_large_cone_mlp_50M", None),
]


def load_task():
    """Load npz splits, drop junk/stranded classes, remap labels densely."""
    name_of = {v: k for k, v in json.load(open(D / "labels.json")).items()}
    raw = {s: np.load(D / f"{s}.npz") for s in ("train", "val", "test")}
    counts = {s: np.bincount(raw[s]["y"], minlength=len(name_of))
              for s in raw}
    keep = [i for i in range(len(name_of))
            if name_of[i] not in DROP_LABELS
            and all(counts[s][i] >= MIN_PER_SPLIT[s] for s in raw)]
    remap = {old: new for new, old in enumerate(keep)}
    task = {}
    for s, z in raw.items():
        sel = np.isin(z["y"], keep)
        task[s] = {"x": z["x"][sel],
                   "y": np.array([remap[v] for v in z["y"][sel]],
                                 dtype=np.int64)}
    classes = [name_of[i] for i in keep]
    print(f"classes kept {len(classes)}/{len(name_of)}:", classes, flush=True)
    for s in task:
        print(s, task[s]["x"].shape, flush=True)
    return task, classes


def normalize(x, nm):
    x = x.copy()
    x[..., 0] = (x[..., 0] - nm.lat_center) / nm.lat_scale
    x[..., 1] = (x[..., 1] - nm.lon_center) / nm.lon_scale
    x[..., 2] = x[..., 2] / nm.sog_scale
    x[..., 5] = x[..., 5] / nm.dt_scale
    return x


def metrics_of(logits, y):
    from sklearn.metrics import accuracy_score, f1_score
    pred = logits.argmax(1)
    top3 = torch.topk(logits, min(3, logits.shape[1]), dim=1).indices
    return {"accuracy": float(accuracy_score(y, pred.numpy())),
            "f1_macro": float(f1_score(y, pred.numpy(), average="macro")),
            "top3_accuracy": float((top3 == torch.from_numpy(y)[:, None])
                                   .any(1).float().mean())}


def torch_probe(emb, ys, n_classes, device):
    """Linear probe: AdamW on standardized features, val-selected LR."""
    from sklearn.metrics import f1_score
    mu = emb["train"].mean(0, keepdims=True)
    sd = emb["train"].std(0, keepdims=True) + 1e-6
    t = {s: torch.from_numpy((emb[s] - mu) / sd).float() for s in emb}
    yt = {s: torch.from_numpy(ys[s]) for s in ys}
    best_state, best_f1, best_lr = None, -1.0, None
    for lr in (3e-3, 1e-3):
        g = torch.Generator().manual_seed(0)
        clf = torch.nn.Linear(t["train"].shape[1], n_classes).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
        n = len(t["train"])
        for epoch in range(15):
            for i in torch.randperm(n, generator=g).split(8192):
                loss = torch.nn.functional.cross_entropy(
                    clf(t["train"][i].to(device)), yt["train"][i].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pv = torch.cat([clf(b.to(device)).cpu()
                                for b in t["val"].split(16384)])
            f1v = f1_score(ys["val"], pv.argmax(1).numpy(), average="macro")
            if f1v > best_f1:
                best_f1, best_lr = f1v, lr
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}
    clf = torch.nn.Linear(t["train"].shape[1], n_classes).to(device)
    clf.load_state_dict(best_state)
    with torch.no_grad():
        pt = torch.cat([clf(b.to(device)).cpu()
                        for b in t["test"].split(16384)])
    return {**metrics_of(pt, ys["test"]), "val_f1_macro": float(best_f1),
            "lr": best_lr}


def main():
    import mlflow
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, classes = load_task()
    ys = {s: task[s]["y"] for s in task}

    base = load_config("configs/pretrain/scaling_large_cone_mlp_50M.yaml",
                       PretrainConfig)
    xn = {s: normalize(task[s]["x"], base.normalization) for s in task}

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("trackfm/ft-vessel-class")
    out = {"classes": classes, "n": {s: int(len(ys[s])) for s in ys}}

    maj = int(np.bincount(ys["train"]).argmax())
    pred = np.full(len(ys["test"]), maj)
    out["baseline-majority-v2"] = {
        "accuracy": float((ys["test"] == maj).mean()),
        "f1_macro": float(f1_score(ys["test"], pred, average="macro"))}

    def kinstats(x):
        sog = x[..., 2]
        dcog = np.abs(np.diff(np.arctan2(x[..., 3], x[..., 4]), axis=1))
        dcog = np.minimum(dcog, 2 * np.pi - dcog)
        return np.stack([sog.mean(1), sog.std(1), np.quantile(sog, .9, 1),
                         dcog.mean(1), dcog.std(1), x[..., 5].mean(1),
                         x[..., 5].std(1)], axis=1)
    out["baseline-kinematic-stats-v2"] = torch_probe(
        {s: kinstats(xn[s]).astype(np.float32) for s in xn}, ys,
        len(classes), device)

    for tag, cfgname, ck in ENCODERS:
        cfg = load_config(f"configs/pretrain/{cfgname}.yaml", PretrainConfig)
        model = build_model(cfg.model, cfg.normalization)
        if ck:
            path = f"/home/paul/data/trackfm/checkpoints/{ck}/best.pt"
            if not Path(path).exists():
                print(f"SKIP {tag}: no checkpoint at {path}", flush=True)
                continue
            sd = torch.load(path, map_location="cpu",
                            weights_only=False)["model"]
            model.load_state_dict(sd)
        model.to(device).eval()
        emb = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            for s in xn:
                parts = []
                for i in range(0, len(xn[s]), 512):
                    e = model.encode(
                        torch.from_numpy(xn[s][i:i + 512]).to(device))
                    parts.append(e.float().mean(dim=1).cpu().numpy())
                emb[s] = np.concatenate(parts)
        del model
        torch.cuda.empty_cache()
        out[tag] = torch_probe(emb, ys, len(classes), device)
        with mlflow.start_run(run_name=f"{tag}-v2-lp"):
            mlflow.log_metrics({f"test_{k}": v for k, v in out[tag].items()
                                if k != "lr"})
            mlflow.log_params({"encoder": tag, "task": "vessel-v2",
                               "n_classes": len(classes),
                               "probe": "torch-linear", "lr": out[tag]["lr"]})
        print(tag, out[tag], flush=True)

    for k in ("baseline-majority-v2", "baseline-kinematic-stats-v2"):
        with mlflow.start_run(run_name=k):
            mlflow.log_metrics({f"test_{m}": v for m, v in out[k].items()
                                if m != "lr"})
            mlflow.log_params({"task": "vessel-v2", "n_classes": len(classes)})
    json.dump(out, open("/home/paul/data/trackfm/ft_vessel_probe_v2.json",
                        "w"), indent=2)
    print("written", flush=True)


if __name__ == "__main__":
    main()
