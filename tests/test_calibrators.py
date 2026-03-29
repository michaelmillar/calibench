from __future__ import annotations

import numpy as np
import pytest

from calibench.calibrators import IsotonicCalibrator, TemperatureScaler
from calibench.metrics import miscalibration_area


def _make_overconfident(n: int = 5_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_std = rng.uniform(1.0, 3.0, size=n)
    predicted_std = true_std * 0.3
    residuals = rng.normal(0.0, true_std)
    return predicted_std, residuals


class TestIsotonicCalibrator:
    def test_reduces_miscalibration(self) -> None:
        predicted_std, residuals = _make_overconfident()
        area_before = miscalibration_area(predicted_std, residuals)

        cal = IsotonicCalibrator()
        calibrated_std = cal.fit_transform(predicted_std, residuals)
        area_after = miscalibration_area(calibrated_std, residuals)

        assert area_after < area_before

    def test_transform_raises_before_fit(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([1.0, 2.0]))

    def test_output_positive(self) -> None:
        predicted_std, residuals = _make_overconfident()
        cal = IsotonicCalibrator()
        calibrated = cal.fit_transform(predicted_std, residuals)
        assert np.all(calibrated > 0)

    def test_fit_returns_self(self) -> None:
        predicted_std, residuals = _make_overconfident(n=100)
        cal = IsotonicCalibrator()
        result = cal.fit(predicted_std, residuals)
        assert result is cal


class TestTemperatureScaler:
    def test_reduces_miscalibration(self) -> None:
        predicted_std, residuals = _make_overconfident()
        area_before = miscalibration_area(predicted_std, residuals)

        scaler = TemperatureScaler()
        calibrated_std = scaler.fit_transform(predicted_std, residuals)
        area_after = miscalibration_area(calibrated_std, residuals)

        assert area_after < area_before

    def test_temperature_greater_than_one_for_overconfident(self) -> None:
        predicted_std, residuals = _make_overconfident()
        scaler = TemperatureScaler()
        scaler.fit(predicted_std, residuals)
        assert scaler.temperature > 1.0

    def test_transform_raises_before_fit(self) -> None:
        scaler = TemperatureScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([1.0, 2.0]))

    def test_fit_returns_self(self) -> None:
        predicted_std, residuals = _make_overconfident(n=100)
        scaler = TemperatureScaler()
        result = scaler.fit(predicted_std, residuals)
        assert result is scaler
