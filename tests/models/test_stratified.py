"""Difficulty-stratified evaluation tests."""
import torch

from trackfm.eval.stratified import difficulty_score, per_sample_soft_ce, stratified_table
from trackfm.training.losses import compute_soft_target_loss


def test_per_sample_ce_matches_mean_loss():
    torch.manual_seed(0)
    ld = torch.log_softmax(torch.randn(8, 3, 16 * 16), dim=-1).reshape(8, 3, 16, 16)
    tgt = (torch.rand(8, 3, 2) - 0.5) * 0.5
    per = per_sample_soft_ce(ld, tgt, 0.3, 16, 0.003)
    assert per.shape == (8, 3)
    mean_ref = compute_soft_target_loss(ld, tgt, 0.3, 16, 0.003)
    torch.testing.assert_close(per.mean(), mean_ref, atol=1e-5, rtol=1e-5)


def test_difficulty_and_deciles():
    torch.manual_seed(1)
    n = 1000
    difficulty = torch.rand(n) * 0.2
    ce = difficulty * 10 + torch.randn(n) * 0.05
    table = stratified_table(difficulty, {"ce_model": ce})
    assert len(table) == 10
    means = [table[d]["ce_model_mean"] for d in range(10)]
    assert means[9] > means[0]
    assert sum(t["n"] for t in table.values()) == n
    lo, hi = table[0]["difficulty_range_deg"], table[9]["difficulty_range_deg"]
    assert lo[0] <= lo[1] <= hi[0] <= hi[1]


def test_difficulty_score():
    dr = torch.zeros(4, 2, 2)
    tgt = torch.tensor([[[0.0, 0.0], [0.3, 0.4]]]).repeat(4, 1, 1)
    d = difficulty_score(dr, tgt)
    torch.testing.assert_close(d[:, 0], torch.zeros(4))
    torch.testing.assert_close(d[:, 1], torch.full((4,), 0.5))
