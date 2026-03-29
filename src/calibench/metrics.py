from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, spearmanr


def miscalibration_area(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    n_bins: int = 20,
) -> float:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    expected_levels = np.linspace(0.0, 1.0, n_bins + 1)[1:]
    observed_coverages = np.array([
        _observed_coverage(predicted_std, residuals, level) for level in expected_levels
    ])

    area = np.trapezoid(np.abs(observed_coverages - expected_levels), expected_levels)
    return float(area)


def expected_calibration_error(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    n_bins: int = 10,
) -> float:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)

    expected_levels = np.linspace(0.0, 1.0, n_bins + 2)[1:-1]
    observed_coverages = np.array([
        _observed_coverage(predicted_std, residuals, level) for level in expected_levels
    ])

    return float(np.mean(np.abs(observed_coverages - expected_levels)))


def sharpness(predicted_std: ArrayLike, level: float = 0.9) -> float:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    z = norm.ppf(0.5 + level / 2.0)
    widths = 2.0 * z * predicted_std
    return float(np.mean(widths))


def coverage_at_confidence(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
    level: float = 0.9,
) -> float:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    return float(_observed_coverage(predicted_std, residuals, level))


def spearman_correlation(
    predicted_std: ArrayLike,
    residuals: ArrayLike,
) -> float:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    abs_residuals = np.abs(residuals)
    result = spearmanr(predicted_std, abs_residuals)
    return float(result.statistic)


def _observed_coverage(
    predicted_std: np.ndarray,
    residuals: np.ndarray,
    level: float,
) -> float:
    z = norm.ppf(0.5 + level / 2.0)
    within = np.abs(residuals) <= z * predicted_std
    return float(np.mean(within))
