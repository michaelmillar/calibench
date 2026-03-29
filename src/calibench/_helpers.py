from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _observed_coverage(
    predicted_std: np.ndarray,
    residuals: np.ndarray,
    level: float,
) -> float:
    z = norm.ppf(0.5 + level / 2.0)
    within = np.abs(residuals) <= z * predicted_std
    return float(np.mean(within))
