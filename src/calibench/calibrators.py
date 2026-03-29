from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(
        self,
        predicted_std: ArrayLike,
        residuals: ArrayLike,
    ) -> "IsotonicCalibrator":
        predicted_std = np.asarray(predicted_std, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)
        abs_residuals = np.abs(residuals)
        self._model.fit(predicted_std, abs_residuals)
        self._fitted = True
        return self

    def transform(self, predicted_std: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted yet")
        predicted_std = np.asarray(predicted_std, dtype=np.float64)
        calibrated = self._model.predict(predicted_std)
        return np.maximum(calibrated, 1e-12)

    def fit_transform(
        self,
        predicted_std: ArrayLike,
        residuals: ArrayLike,
    ) -> np.ndarray:
        self.fit(predicted_std, residuals)
        return self.transform(predicted_std)


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._fitted = False

    def fit(
        self,
        predicted_std: ArrayLike,
        residuals: ArrayLike,
    ) -> "TemperatureScaler":
        predicted_std = np.asarray(predicted_std, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)

        def neg_log_likelihood(log_t: float) -> float:
            t = np.exp(log_t)
            scaled_std = predicted_std * t
            nll = 0.5 * np.mean(
                np.log(2.0 * np.pi * scaled_std**2) + (residuals / scaled_std) ** 2
            )
            return float(nll)

        result = minimize_scalar(neg_log_likelihood, bounds=(-5.0, 5.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        self._fitted = True
        return self

    def transform(self, predicted_std: ArrayLike) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fitted yet")
        predicted_std = np.asarray(predicted_std, dtype=np.float64)
        return predicted_std * self.temperature

    def fit_transform(
        self,
        predicted_std: ArrayLike,
        residuals: ArrayLike,
    ) -> np.ndarray:
        self.fit(predicted_std, residuals)
        return self.transform(predicted_std)
