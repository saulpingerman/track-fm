"""Build vessel-class-v2: ship-type classification from our own corpus.

Replaces exp-13's 4-class/2.2k-track task (too small to rank encoders —
FT-sweep verdict) with 12+ AIS ship-type classes over hundreds of
thousands of windows. Sources: v3sub200 VAL-period windows (meta carries
mmsi; val period avoids pretrain-train overlap) joined with the
mmsi->Ship-type table extracted from the 26 monthly raw archives.

Splits are VESSEL-DISJOINT (mmsi hash 70/15/15) so no identity leakage.
Windows = the 128-posit input section, converted from v3 raw columns
[lat,lon,sog,cog_deg,dt,heading] to the model format
[lat,lon,sog,cog_sin,cog_cos,dt] in raw units (probe scripts normalize).

Usage: python scripts/build_vessel_v2.py
Writes ~/data/trackfm/vessel_v2/{train,val,test}.npz + labels.json + MANIFEST.json
"""
from __future__ import annotations

import glob
import json
import hashlib

import numpy as np
import pyarrow.parquet as pq

TABLE = json.load(open("/home/paul/data/trackfm/mmsi_shiptype.json"))
SHARDS = sorted(glob.glob("/home/paul/data/trackfm/materialized/v3sub200/val/*.parquet"))
OUT = "/home/paul/data/trackfm/vessel_v2"
MIN_CLASS = 500
CAP_PER_CLASS_SPLIT = {"train": 35000, "val": 7500, "test": 7500}
DROP = {"Undefined", "Other", "Reserved", "Spare 1", "Spare 2"}


def split_of(mmsi: int) -> str:
    h = int(hashlib.md5(str(mmsi).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    # pass 1: class counts on a metadata-only sweep (cheap: meta = 4 floats,
    # but they live inside the packed vector — stream and slice)
    buf = {s: {"x": [], "y": [], "m": []} for s in ("train", "val", "test")}
    counts = {}
    label_of = {}
    fill = {}          # (split, class) -> count, honoring caps

    for sp in SHARDS:
        pf = pq.ParquetFile(sp)
        for batch in pf.iter_batches(batch_size=8192, columns=["features"]):
            a = np.stack(batch.column("features").to_numpy(zero_copy_only=False))
            meta, body = a[:, :4], a[:, 4:]
            mmsi = (meta[:, 2].astype(np.int64) * 100_000
                    + meta[:, 3].astype(np.int64))
            for i, m in enumerate(mmsi):
                t = TABLE.get(str(m))
                if t is None or t in DROP:
                    continue
                s = split_of(int(m))
                key = (s, t)
                if fill.get(key, 0) >= CAP_PER_CLASS_SPLIT[s]:
                    continue
                w = body[i].reshape(928, 6)[:128]        # input section, raw
                cog = np.deg2rad(w[:, 3])
                x = np.stack([w[:, 0], w[:, 1], w[:, 2],
                              np.sin(cog), np.cos(cog), w[:, 4]], axis=1)
                buf[s]["x"].append(x.astype(np.float32))
                buf[s]["y"].append(t)
                buf[s]["m"].append(int(m))
                fill[key] = fill.get(key, 0) + 1
                counts[t] = counts.get(t, 0) + 1
        print(sp.split("/")[-1], {k: len(v["y"]) for k, v in buf.items()},
              flush=True)

    keep = sorted([c for c, n in counts.items() if n >= MIN_CLASS])
    label_of = {c: i for i, c in enumerate(keep)}
    print("classes kept:", keep, flush=True)
    manifest = {"classes": keep, "counts": counts,
                "caps": CAP_PER_CLASS_SPLIT, "min_class": MIN_CLASS,
                "source": "v3sub200/val x mmsi_shiptype (26 monthly zips)",
                "splits": "vessel-disjoint md5(mmsi) 70/15/15",
                "feature_format": "[lat,lon,sog,cog_sin,cog_cos,dt] raw units"}
    for s in ("train", "val", "test"):
        sel = [i for i, t in enumerate(buf[s]["y"]) if t in label_of]
        x = np.stack([buf[s]["x"][i] for i in sel])
        y = np.array([label_of[buf[s]["y"][i]] for i in sel], dtype=np.int64)
        m = np.array([buf[s]["m"][i] for i in sel], dtype=np.int64)
        np.savez_compressed(f"{OUT}/{s}.npz", x=x, y=y, mmsi=m)
        manifest[f"n_{s}"] = int(len(y))
        print(s, x.shape, "classes", len(np.unique(y)), flush=True)
    json.dump(manifest, open(f"{OUT}/MANIFEST.json", "w"), indent=2)
    json.dump(label_of, open(f"{OUT}/labels.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
