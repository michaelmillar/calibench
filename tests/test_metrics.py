from __future__ import annotations

import numpy as np

from calibench.metrics import (
    coverage_at_confidence,
    expected_calibration_error,
    miscalibration_area,
    sharpness,
    spearman_correlation,
)


def _make_calibrated(n: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predicted_std = rng.uniform(0.5, 2.0, size=n)
    residuals = rng.normal(0.0, predicted_std)
    return predicted_std, residuals


def _make_overconfident(n: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_std = rng.uniform(1.0, 3.0, size=n)
    predicted_std = true_std * 0.3
    residuals = rng.normal(0.0, true_std)
    return predicted_std, residuals


class TestMiscalibrationArea:
    def test_perfect_calibration_near_zero(self) -> None:
        predicted_std, residuals = _make_calibrated()
        area = miscalibration_area(predicted_std, residuals)
        assert area < 0.02

    def test_overconfident_has_larger_area(self) -> None:
        cal_std, cal_res = _make_calibrated()
        ov_std, ov_res = _make_overconfident()
        area_cal = miscalibration_area(cal_std, cal_res)
        area_ov = miscalibration_area(ov_std, ov_res)
        assert area_ov > area_cal * 5


class TestExpectedCalibrationError:
    def test_perfect_calibration_near_zero(self) -> None:
        predicted_std, residuals = _make_calibrated()
        ece = expected_calibration_error(predicted_std, residuals)
        assert ece < 0.03

    def test_overconfident_has_larger_ece(self) -> None:
        cal_std, cal_res = _make_calibrated()
        ov_std, ov_res = _make_overconfident()
        ece_cal = expected_calibration_error(cal_std, cal_res)
        ece_ov = expected_calibration_error(ov_std, ov_res)
        assert ece_ov > ece_cal * 3


class TestSharpness:
    def test_sharpness_positive(self) -> None:
        predicted_std = np.array([1.0, 2.0, 3.0])
        s = sharpness(predicted_std, level=0.9)
        assert s > 0

    def test_sharpness_scales_with_std(self) -> None:
        s1 = sharpness(np.array([1.0, 1.0]), level=0.9)
        s2 = sharpness(np.array([2.0, 2.0]), level=0.9)
        assert abs(s2 / s1 - 2.0) < 0.01


class TestCoverageAtConfidence:
    def test_calibrated_coverage_near_target(self) -> None:
        predicted_std, residuals = _make_calibrated(n=50_000)
        cov = coverage_at_confidence(predicted_std, residuals, level=0.9)
        assert abs(cov - 0.9) < 0.02

    def test_overconfident_coverage_below_target(self) -> None:
        predicted_std, residuals = _make_overconfident()
        cov = coverage_at_confidence(predicted_std, residuals, level=0.9)
        assert cov < 0.5


class TestSpearmanCorrelation:
    def test_positive_correlation_for_calibrated(self) -> None:
        predicted_std, residuals = _make_calibrated()
        corr = spearman_correlation(predicted_std, residuals)
        assert corr > 0.3

    def test_returns_float(self) -> None:
        predicted_std, residuals = _make_calibrated(n=100)
        corr = spearman_correlation(predicted_std, residuals)
        assert isinstance(corr, float)
