from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from calibench.visualise import (
    calibration_curve,
    reliability_diagram,
    uncertainty_vs_error_scatter,
)


def _make_calibrated(n: int = 1_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predicted_std = rng.uniform(0.5, 2.0, size=n)
    residuals = rng.normal(0.0, predicted_std)
    return predicted_std, residuals


class TestReliabilityDiagram:
    def test_returns_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        ax = reliability_diagram(predicted_std, residuals)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        fig, ax = plt.subplots()
        returned_ax = reliability_diagram(predicted_std, residuals, ax=ax)
        assert returned_ax is ax
        plt.close("all")


class TestCalibrationCurve:
    def test_returns_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        ax = calibration_curve(predicted_std, residuals)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        fig, ax = plt.subplots()
        returned_ax = calibration_curve(predicted_std, residuals, ax=ax)
        assert returned_ax is ax
        plt.close("all")


class TestUncertaintyVsErrorScatter:
    def test_returns_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        ax = uncertainty_vs_error_scatter(predicted_std, residuals)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_accepts_existing_axes(self) -> None:
        predicted_std, residuals = _make_calibrated()
        fig, ax = plt.subplots()
        returned_ax = uncertainty_vs_error_scatter(predicted_std, residuals, ax=ax)
        assert returned_ax is ax
        plt.close("all")
