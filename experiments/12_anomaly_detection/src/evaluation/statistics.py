"""
Statistical analysis for experiment results.

Includes bootstrap confidence intervals and paired statistical tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from collections import defaultdict


def aggregate_results(
    all_results: Dict[str, List[Dict]]
) -> pd.DataFrame:
    """
    Aggregate results across folds and seeds.

    Args:
        all_results: Dictionary mapping condition -> list of fold results

    Returns:
        DataFrame with mean ± std for each metric
    """
    rows = []

    for condition, results in all_results.items():
        # Extract test metrics from each fold
        metrics_list = [r['test_metrics'] for r in results]

        # Compute mean and std for each metric
        metric_names = metrics_list[0].keys()

        row = {'condition': condition}
        for metric in metric_names:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_std'] = std_val
                row[metric] = f"{mean_val:.3f}±{std_val:.3f}"

        rows.append(row)

    return pd.DataFrame(rows)


def run_statistical_tests(
    all_results: Dict[str, List[Dict]],
    config: Dict
) -> pd.DataFrame:
    """
    Run statistical tests comparing conditions.

    Uses paired t-test and Wilcoxon signed-rank test.

    Args:
        all_results: Dictionary mapping condition -> list of fold results
        config: Experiment configuration

    Returns:
        DataFrame with test results
    """
    alpha = config['evaluation']['significance_level']
    conditions = list(all_results.keys())
    metrics = ['auroc', 'auprc', 'f1_optimal']

    rows = []

    # Compare each pair of conditions
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            for metric in metrics:
                # Get paired values
                values1 = [r['test_metrics'][metric] for r in all_results[cond1]]
                values2 = [r['test_metrics'][metric] for r in all_results[cond2]]

                # Ensure paired comparison (same folds)
                min_len = min(len(values1), len(values2))
                values1 = values1[:min_len]
                values2 = values2[:min_len]

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(values1, values2)

                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_pval = stats.wilcoxon(values1, values2)
                except ValueError:
                    # Wilcoxon fails if all differences are zero
                    w_stat, w_pval = 0, 1.0

                # Effect size (Cohen's d for paired samples)
                diff = np.array(values1) - np.array(values2)
                effect_size = np.mean(diff) / (np.std(diff) + 1e-10)

                rows.append({
                    'comparison': f'{cond1} vs {cond2}',
                    'metric': metric,
                    'mean_diff': np.mean(diff),
                    'std_diff': np.std(diff),
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'wilcoxon_pvalue': w_pval,
                    'effect_size': effect_size,
                    'significant': t_pval < alpha
                })

    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    statistic: str = 'mean'
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Array of values
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        statistic: Statistic to compute ('mean', 'median')

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n = len(values)

    if statistic == 'mean':
        stat_fn = np.mean
    elif statistic == 'median':
        stat_fn = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    point_estimate = stat_fn(values)

    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats[i] = stat_fn(sample)

    # Percentile confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def compute_metric_cis(
    all_results: Dict[str, List[Dict]],
    n_bootstrap: int = 10000
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for all metrics.

    Args:
        all_results: Dictionary mapping condition -> list of fold results
        n_bootstrap: Number of bootstrap samples

    Returns:
        DataFrame with CIs for each condition and metric
    """
    rows = []
    metrics = ['auroc', 'auprc', 'f1_optimal']

    for condition, results in all_results.items():
        for metric in metrics:
            values = np.array([r['test_metrics'][metric] for r in results])
            mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap)

            rows.append({
                'condition': condition,
                'metric': metric,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower
            })

    return pd.DataFrame(rows)


def paired_bootstrap_test(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    Paired bootstrap test for difference in means.

    Args:
        values1: First set of values
        values2: Second set of values (paired with values1)
        n_bootstrap: Number of bootstrap samples

    Returns:
        (mean_difference, p_value)
    """
    diff = values1 - values2
    observed_diff = np.mean(diff)

    # Bootstrap under null hypothesis (difference centered at 0)
    centered_diff = diff - np.mean(diff)

    n = len(diff)
    bootstrap_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(centered_diff, size=n, replace=True)
        bootstrap_diffs[i] = np.mean(sample)

    # Two-tailed p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value


def summarize_by_condition(
    all_results: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    """
    Create summary statistics for each condition.

    Args:
        all_results: Dictionary mapping condition -> list of fold results

    Returns:
        Dictionary with summary for each condition
    """
    summary = {}

    for condition, results in all_results.items():
        metrics_list = [r['test_metrics'] for r in results]

        condition_summary = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            condition_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        summary[condition] = condition_summary

    return summary
