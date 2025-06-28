print('hello')

#!/usr/bin/env python
# train_seq20_f128.py  –  headless overnight training script
# -----------------------------------------------------------
import math, os, sys, time
from pathlib import Path
from typing import Optional

import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import IterableDataset, DataLoader

# ─── make sure Fourier head repo is importable ────────────────────────────
fourier_dir = Path.home() / "repos" / "fourier-head" / "notebooks"
sys.path.insert(0, str(fourier_dir))
from four_head_2D import FourierHead2D          # full Fourier head

# ─── geometry helper (same as notebook, Jacobian sign fixed) ──────────────
def latlon_to_local_uv(lat, lon, lat0, lon0, half_side_mi: float = 50.0):
    R_mi = 69.0
    dx = R_mi * torch.cos(torch.deg2rad(lat0)) * (lon - lon0)
    dy = R_mi * (lat - lat0)
    u  = dx / half_side_mi
    v  = dy / half_side_mi
    logJ = (
        2.0 * math.log(half_side_mi)
        - 2.0 * math.log(R_mi)
        - torch.log(torch.cos(torch.deg2rad(lat0)).clamp_min(1e-6))
    )
    uv = torch.stack([u.clamp(-1, 1), v.clamp(-1, 1)], dim=-1)
    return uv, logJ

# ─── Polars loader ────────────────────────────────────────────────────────
def load_cleaned_data(root: str) -> pl.DataFrame:
    files = list(Path(root).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No parquet files under {}".format(root))
    print(f"🟢 Loading {len(files)} parquet file(s)")
    return pl.read_parquet([str(f) for f in files])

# ──────────────────────────────────────────────────────────────────────────
#  Fast, RAM-buffered streaming dataset
# ──────────────────────────────────────────────────────────────────────────
#
#  • Scans the Polars DataFrame ONCE, groups by MMSI, and materialises each
#    vessel track as a torch tensor in memory.  Iteration thereafter is pure
#    Python/Torch—no repeated DataFrame filtering, so the first batch
#    arrives quickly even with thousands of vessels.
#  • Yields the exact tuple your training loop expects:
#      (past_uv, centre_ll_norm, target_uv, logJ)
#
#  Requires:  latlon_to_local_uv(...) helper already defined.
# --------------------------------------------------------------------------
from torch.utils.data import IterableDataset
import polars as pl
import torch

class AISForecastIterableDataset(IterableDataset):
    """
    Streams windows one vessel at a time.

    Yields
    ------
    past_uv : (seq_len, 2) tensor            # local coords in [-1,1]²
    centre_ll_norm : (2,) tensor             # absolute context (lat/90 , lon/180)
    target_uv : (2,) tensor                  # next point in local [-1,1]²
    logJ : scalar tensor                     # log Jacobian for the window
    """
    def __init__(self,
                 df: pl.DataFrame,
                 seq_len: int       = 20,
                 half_side_mi: float = 50.0):
        print(">>>     BUFFERED DATASET VERSION LOADED     <<<", flush=True)
        self.seq_len = seq_len
        self.half    = half_side_mi

        # 1) sort once, then group once
        df_sorted = df.sort(["mmsi", "timestamp"])

        grouped = (
            df_sorted
            .group_by("mmsi")
            .agg(pl.struct(["lat", "lon"]).alias("track"))
        )

        # 2) materialise each vessel track as a tensor
        self.tracks = []
        for row in grouped.iter_rows(named=True):
            coords_np = row["track"].to_numpy()          # (N,2) float64
            if coords_np.shape[0] > seq_len + 1:
                self.tracks.append(
                    torch.tensor(coords_np, dtype=torch.float32)
                )

        print(f"🟢 Dataset ready — {len(self.tracks)} vessels buffered in RAM",
              flush=True)

    # ---------------------------------------------------------------------
    def __iter__(self):
        for coords in self.tracks:               # coords : (N,2) tensor
            N = coords.size(0)
            for i in range(self.seq_len, N - 1):
                past_abs   = coords[i - self.seq_len : i]   # (S,2)
                target_abs = coords[i]                      # (2,)

                lat0, lon0 = past_abs[-1]                   # window centre

                past_uv, _      = latlon_to_local_uv(
                    past_abs[:, 0], past_abs[:, 1],
                    lat0, lon0, self.half
                )
                target_uv, logJ = latlon_to_local_uv(
                    target_abs[0], target_abs[1],
                    lat0, lon0, self.half
                )

                centre_ll_norm = torch.tensor(
                    [lat0 / 90.0, lon0 / 180.0],
                    dtype=torch.float32
                )

                yield past_uv, centre_ll_norm, target_uv, logJ


# ─── Model ────────────────────────────────────────────────────────────────
class TransformerForecaster(nn.Module):
    def __init__(self, seq_len, d_model, nhead, num_layers,
                 ff_hidden, fourier_m):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff_hidden,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.fh      = FourierHead2D(dim_input=d_model,
                                     num_frequencies=fourier_m)

    def forward(self, past_uv, centre_ll, target_uv):
        B, S, _ = past_uv.shape
        x = torch.cat([past_uv,
                       centre_ll.unsqueeze(1).expand(-1,S,-1)], dim=-1)
        h = self.input_proj(x) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        last = h[:, -1]
        return self.fh(last, target_uv)

# ─── Training loop ────────────────────────────────────────────────────────
def train(df: pl.DataFrame,
          seq_len=20,
          d_model=128,
          nhead=4,
          num_layers=4,
          ff_hidden=512,
          fourier_m=128,
          batch_size=64,
          lr=1e-5,
          epochs=100,
          ckpt_dir="checkpoints",
          model_tag="seq20_f128",
          ckpt_every=10_000,
          device: Optional[str]=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = AISForecastIterableDataset(df, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=True, num_workers=0, pin_memory=True)

    model = TransformerForecaster(seq_len, d_model, nhead,
                                  num_layers, ff_hidden, fourier_m).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    Path(ckpt_dir, model_tag).mkdir(parents=True, exist_ok=True)
    global_step = 0

    for ep in range(1, epochs+1):
        total, count = 0.0, 0
        start = time.time()
        for batch_i, (x, c, y, logJ) in enumerate(loader, 1):
            global_step += 1
            x, c, y, logJ = x.to(device), c.to(device), y.to(device), logJ.to(device)
            pdf = model(x, c, y)
            loss = -(pdf.log() + logJ).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0);  count += x.size(0)

            if global_step % ckpt_every == 0:
                ckpt_path = Path(ckpt_dir, model_tag,
                                 f"ep{ep:02d}_step{global_step:07d}.pt")
                torch.save({"epoch": ep,
                            "step": global_step,
                            "model_state": model.state_dict(),
                            "opt_state":   opt.state_dict()},
                           ckpt_path)
                print(f"💾 checkpoint → {ckpt_path}")

        avg = total / count
        t   = time.time() - start
            print(f"🟢 Epoch {ep}/{epochs} | NLL {avg:.4f} | {t/60:.1f} min")

    return model

# ─── main guard ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_cleaned_data("data/cleaned_partitioned_ais")
    train(df,
          seq_len=20,
          d_model=128,
          nhead=4,
          num_layers=4,
          ff_hidden=512,
          fourier_m=128,
          batch_size=64,
          lr=1e-5,
          epochs=100,
          ckpt_dir="checkpoints",
          model_tag="seq20_f128")
    print("✅ Training complete")
