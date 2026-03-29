from __future__ import annotations

import dataclasses

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from calibench.calibrators import TemperatureScaler
from calibench.metrics import (
    coverage_at_confidence,
    expected_calibration_error,
    miscalibration_area,
    sharpness,
    spearman_correlation,
)
from calibench.visualise import (
    calibration_curve,
    reliability_diagram,
    uncertainty_vs_error_scatter,
)


def _compute_verdict(coverage_at_90: float) -> str:
    if coverage_at_90 < 0.85:
        return "overconfident"
    if coverage_at_90 > 0.95:
        return "underconfident"
    return "well-calibrated"


def _run_metrics(
    predicted_std: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(predicted_std, residuals),
        "miscalibration_area": miscalibration_area(predicted_std, residuals),
        "sharpness": sharpness(predicted_std),
        "coverage_at_90": coverage_at_confidence(predicted_std, residuals, level=0.9),
        "spearman_correlation": spearman_correlation(predicted_std, residuals),
    }


@dataclasses.dataclass(frozen=True)
class Report:
    ece: float
    miscalibration_area: float
    sharpness: float
    coverage_at_90: float
    spearman_correlation: float

    recalibrated_ece: float
    recalibrated_miscalibration_area: float
    recalibrated_sharpness: float
    recalibrated_coverage_at_90: float
    recalibrated_spearman_correlation: float

    verdict: str
    temperature: float
    calibrator: TemperatureScaler = dataclasses.field(repr=False)

    predicted_std: np.ndarray = dataclasses.field(repr=False)
    residuals: np.ndarray = dataclasses.field(repr=False)
    recalibrated_std: np.ndarray = dataclasses.field(repr=False)

    def plot(self) -> matplotlib.figure.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        reliability_diagram(self.predicted_std, self.residuals, ax=axes[0])
        calibration_curve(self.predicted_std, self.residuals, ax=axes[1])
        uncertainty_vs_error_scatter(self.predicted_std, self.residuals, ax=axes[2])
        fig.suptitle(f"Calibration audit  {self.verdict}")
        fig.tight_layout()
        return fig

    def to_dict(self) -> dict[str, float | str]:
        result: dict[str, float | str] = {}
        for f in dataclasses.fields(self):
            if f.type in ("np.ndarray", "TemperatureScaler"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, (int, float, str)):
                result[f.name] = val
        return result

    def to_markdown(self) -> str:
        rows = [
            ("ECE", self.ece, self.recalibrated_ece),
            (
                "Miscalibration area",
                self.miscalibration_area,
                self.recalibrated_miscalibration_area,
            ),
            ("Sharpness", self.sharpness, self.recalibrated_sharpness),
            ("Coverage @ 90%", self.coverage_at_90, self.recalibrated_coverage_at_90),
            ("Spearman r", self.spearman_correlation, self.recalibrated_spearman_correlation),
        ]
        lines = [
            "## calibench audit",
            "",
            f"**Verdict:** {self.verdict} (temperature = {self.temperature:.2f})",
            "",
            "| Metric | Before | After |",
            "|---|---|---|",
        ]
        for name, before, after in rows:
            lines.append(f"| {name} | {before:.4f} | {after:.4f} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        rows = [
            ("ECE", self.ece, self.recalibrated_ece),
            (
                "Miscalibration area",
                self.miscalibration_area,
                self.recalibrated_miscalibration_area,
            ),
            ("Sharpness", self.sharpness, self.recalibrated_sharpness),
            ("Coverage @ 90%", self.coverage_at_90, self.recalibrated_coverage_at_90),
            ("Spearman r", self.spearman_correlation, self.recalibrated_spearman_correlation),
        ]
        header = "calibench audit"
        lines = [
            header,
            "=" * len(header),
            f"Verdict: {self.verdict} (temperature = {self.temperature:.2f})",
            "",
            f"{'':20s}  {'Before':>10s}  {'After':>10s}",
        ]
        for name, before, after in rows:
            lines.append(f"{name:20s}  {before:10.4f}  {after:10.4f}")
        return "\n".join(lines)


def audit(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_std: ArrayLike,
) -> Report:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)

    residuals = y_true - y_pred

    before = _run_metrics(y_std, residuals)

    scaler = TemperatureScaler()
    recalibrated_std = scaler.fit_transform(y_std, residuals)

    after = _run_metrics(recalibrated_std, residuals)

    return Report(
        ece=before["ece"],
        miscalibration_area=before["miscalibration_area"],
        sharpness=before["sharpness"],
        coverage_at_90=before["coverage_at_90"],
        spearman_correlation=before["spearman_correlation"],
        recalibrated_ece=after["ece"],
        recalibrated_miscalibration_area=after["miscalibration_area"],
        recalibrated_sharpness=after["sharpness"],
        recalibrated_coverage_at_90=after["coverage_at_90"],
        recalibrated_spearman_correlation=after["spearman_correlation"],
        verdict=_compute_verdict(before["coverage_at_90"]),
        temperature=scaler.temperature,
        calibrator=scaler,
        predicted_std=y_std,
        residuals=residuals,
        recalibrated_std=recalibrated_std,
    )
