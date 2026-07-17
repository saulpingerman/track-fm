"""Soft-target aliasing at G=64: how much loss floor + gradient noise does
sigma = 0.32 cells create, and would a wider sigma help?

Mechanism: the soft target is a Gaussian discretized on the G x G grid.
With sigma much smaller than a cell, the discretized target's SHAPE
depends on where the true position falls WITHIN its cell (center: all
mass in one cell; corner: split 4 ways). The model cannot see sub-cell
position, so its best possible prediction p* is the average target over
sub-cell offsets — and the loss it pays, E_offset[ CE(q_offset, p*) ],
exceeds the entropy of any single target. That excess is pure aliasing:
irreducible, position-dependent loss noise that dilutes the training
signal.

This script quantifies, per sigma (in cells):
  floor_ce   = E_offset[ CE(q_off, p*) ]  with p* = E[q_off]  (best case)
  ce_spread  = std over offsets of CE(q_off, p*)   (per-sample loss noise)
  excess     = floor_ce - E[H(q_off)]              (pure aliasing penalty)
  blur_cells = std of p* in cells                  (supervision blur cost)
"""
import numpy as np

G = 15                        # local window is enough (sigma <= 2 cells)
K = 33                        # sub-cell offset grid (K x K offsets per cell)


def discretized_target(sigma_cells: float, ox: float, oy: float) -> np.ndarray:
    """Gaussian at cell-center offset (ox, oy) in [-0.5, 0.5), discretized."""
    c = np.arange(G) - G // 2
    xx, yy = np.meshgrid(c, c, indexing="ij")
    d2 = (xx - ox) ** 2 + (yy - oy) ** 2
    q = np.exp(-d2 / (2 * sigma_cells ** 2))
    return q / q.sum()


def analyze(sigma_cells: float) -> dict:
    offs = np.linspace(-0.5, 0.5, K, endpoint=False)
    qs = np.stack([discretized_target(sigma_cells, ox, oy)
                   for ox in offs for oy in offs])
    p_star = qs.mean(0)
    p_star = p_star / p_star.sum()
    logp = np.log(np.clip(p_star, 1e-30, None))
    ces = -(qs * logp).sum(axis=(1, 2))
    ents = -(qs * np.log(np.clip(qs, 1e-30, None))).sum(axis=(1, 2))
    # blur: 1-sigma radius of p*
    c = np.arange(G) - G // 2
    xx, yy = np.meshgrid(c, c, indexing="ij")
    var = (p_star * (xx ** 2 + yy ** 2)).sum() - (
        (p_star * xx).sum() ** 2 + (p_star * yy).sum() ** 2)
    return dict(sigma_cells=sigma_cells,
                floor_ce=float(ces.mean()),
                ce_spread=float(ces.std()),
                mean_entropy=float(ents.mean()),
                excess=float(ces.mean() - ents.mean()),
                blur_cells=float(np.sqrt(max(var, 0))))


print(f"{'sig_cells':>9} {'floor_CE':>9} {'CE_noise':>9} {'H(q)':>7} "
      f"{'aliasing':>9} {'blur':>6}")
rows = []
for s in (0.32, 0.5, 0.75, 1.0, 1.5):
    r = analyze(s)
    rows.append(r)
    print(f"{r['sigma_cells']:>9.2f} {r['floor_ce']:>9.4f} {r['ce_spread']:>9.4f} "
          f"{r['mean_entropy']:>7.4f} {r['excess']:>9.4f} {r['blur_cells']:>6.2f}")

# context: current small-cone val_loss ~1.26; how big is the aliasing
# penalty relative to the loss scale?
cur = rows[0]
print(f"\nAt the CURRENT sigma (0.32 cells): aliasing penalty = "
      f"{cur['excess']:.3f} nats with +-{cur['ce_spread']:.3f} per-sample noise.")
print(f"Best-trained val_loss ~1.26 (small cone) — aliasing is "
      f"{cur['excess']/1.26*100:.0f}% of it and the noise is "
      f"{cur['ce_spread']/1.26*100:.0f}%.")
