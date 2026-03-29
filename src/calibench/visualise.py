from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def reliability_diagram(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    n_bins: int = 20,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    expected_levels = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    observed = np.array([
        _observed_coverage(predicted_std, residuals, lvl) for lvl in expected_levels
    ])

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Ideal")
    ax.plot(expected_levels, observed, marker="o", color="#1f77b4", linewidth=2, label="Observed")
    ax.fill_between(
        expected_levels,
        expected_levels,
        observed,
        alpha=0.25,
        color="#ff7f0e",
    )
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    return ax


def calibration_curve(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    n_bins: int = 20,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    abs_residuals = np.abs(residuals)

    sorted_idx = np.argsort(predicted_std)
    sorted_std = predicted_std[sorted_idx]
    sorted_abs = abs_residuals[sorted_idx]

    bin_edges = np.array_split(np.arange(len(sorted_std)), n_bins)
    mean_std = np.array([sorted_std[idx].mean() for idx in bin_edges])
    mean_abs = np.array([sorted_abs[idx].mean() for idx in bin_edges])

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    max_val = max(mean_std.max(), mean_abs.max()) * 1.1
    ax.plot(
        [0, max_val], [0, max_val],
        linestyle="--", color="#999999", linewidth=1, label="Ideal",
    )
    ax.plot(mean_std, mean_abs, marker="o", color="#1f77b4", linewidth=2, label="Observed")
    ax.set_xlabel("Mean predicted std")
    ax.set_ylabel("Mean absolute residual")
    ax.set_title("Calibration curve")
    ax.legend(loc="lower right")

    return ax


def uncertainty_vs_error_scatter(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    ax: matplotlib.axes.Axes | None = None,
    alpha: float = 0.3,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    abs_residuals = np.abs(residuals)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(predicted_std, abs_residuals, alpha=alpha, s=10, color="#1f77b4", **kwargs)

    max_val = max(predicted_std.max(), abs_residuals.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#ff7f0e", linewidth=1, label="y=x")
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("|Residual|")
    ax.set_title("Uncertainty vs error")
    ax.legend(loc="upper left")

    return ax


def _observed_coverage(
    predicted_std: np.ndarray,
    residuals: np.ndarray,
    level: float,
) -> float:
    z = norm.ppf(0.5 + level / 2.0)
    within = np.abs(residuals) <= z * predicted_std
    return float(np.mean(within))
