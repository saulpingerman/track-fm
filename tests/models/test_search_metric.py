"""Search-effort metric tests."""
import torch

from trackfm.eval.search import gaussian_grid_log_density, search_ranks, summarize_ranks

G, R = 32, 0.3
CELL = 2 * R / G


def peaked_density(cx_idx, cy_idx):
    """Density concentrated at one grid cell."""
    d = torch.full((1, 1, G, G), 1e-12)
    d[0, 0, cx_idx, cy_idx] = 1.0
    d = d / d.sum(dim=(-2, -1), keepdim=True)
    return d.log()


def cell_center(i):
    return -R + (i + 0.5) * CELL


def test_perfect_prediction_rank_1():
    ld = peaked_density(10, 20)
    tgt = torch.tensor([[[cell_center(10), cell_center(20)]]])
    r = search_ranks(ld, tgt, R)
    assert r.item() == 1


def test_rank_counts_higher_probability_cells():
    d = torch.full((1, 1, G, G), 1e-12)
    d[0, 0, 5, 5] = 0.5     # wrong guess, highest
    d[0, 0, 6, 6] = 0.3     # wrong, second
    d[0, 0, 7, 7] = 0.2     # truth here, third
    ld = (d / d.sum(dim=(-2, -1), keepdim=True)).log()
    tgt = torch.tensor([[[cell_center(7), cell_center(7)]]])
    assert search_ranks(ld, tgt, R).item() == 3


def test_off_grid_target_censored():
    ld = peaked_density(0, 0)
    tgt = torch.tensor([[[0.5, 0.0]]])  # beyond +/-0.3
    r = search_ranks(ld, tgt, R)
    assert r.item() == -1
    s = summarize_ranks(r)
    assert s["censored_frac"] == 1.0


def test_footprint_aggregation_reduces_rank():
    """Coarser footprints can only make capture easier (rank <= fine rank)."""
    torch.manual_seed(0)
    ld = torch.log_softmax(torch.randn(4, 2, G * G), dim=-1).reshape(4, 2, G, G)
    tgt = (torch.rand(4, 2, 2) - 0.5) * 2 * R * 0.9
    fine = search_ranks(ld, tgt, R, footprint_cells=1)
    coarse = search_ranks(ld, tgt, R, footprint_cells=4)
    assert (coarse[fine > 0] <= fine[fine > 0]).all()


def test_sharper_gaussian_ranks_better_when_accurate():
    """A tight Gaussian at the right spot needs fewer images than a wide one."""
    pred = torch.zeros(1, 1, 2)
    tgt = torch.tensor([[[CELL * 0.4, CELL * 0.4]]])  # within the center cell
    tight = gaussian_grid_log_density(pred, torch.full((1, 1, 2), 0.01), R, G)
    wide = gaussian_grid_log_density(pred, torch.full((1, 1, 2), 0.15), R, G)
    rt = search_ranks(tight, tgt, R).item()
    rw = search_ranks(wide, tgt, R).item()
    assert rt <= rw
    assert rt <= 4


def test_capture_curve_properties():
    from trackfm.eval.search import capture_curve

    torch.manual_seed(1)
    N = G * G
    ld = torch.log_softmax(torch.randn(64, 1, G * G) * 3, dim=-1).reshape(64, 1, G, G)
    tgt = (torch.rand(64, 1, 2) - 0.5) * 2 * R * 0.9
    ranks = search_ranks(ld, tgt, R)
    c = capture_curve(ranks, N)
    cap = torch.tensor(c["capture"])
    assert (cap.diff() >= 0).all()                      # monotone
    assert abs(cap[-1].item() - c["ceiling"]) < 1e-6    # asymptotes at ceiling
    assert 0 <= c["auc"] <= 1
    if c["k@90"] is not None:
        assert cap[c["k@90"] - 1] >= 0.9


def test_capture_curve_perfect_model_beats_null():
    from trackfm.eval.search import capture_curve

    # density peaked exactly at truth for every sample -> rank 1 always
    tgt = torch.zeros(8, 1, 2) + CELL * 0.4
    ld = peaked_density(G // 2, G // 2).repeat(8, 1, 1, 1)
    ranks = search_ranks(ld, tgt, R)
    c = capture_curve(ranks, G * G)
    assert c["k@90"] == 1
    assert c["auc"] > c["null_auc"]
