from calibench.calibrators import IsotonicCalibrator, TemperatureScaler
from calibench.metrics import (
    coverage_at_confidence,
    expected_calibration_error,
    miscalibration_area,
    sharpness,
    spearman_correlation,
)
from calibench.report import Report, audit
from calibench.visualise import (
    calibration_curve,
    reliability_diagram,
    uncertainty_vs_error_scatter,
)

__all__ = [
    "audit",
    "Report",
    "coverage_at_confidence",
    "expected_calibration_error",
    "miscalibration_area",
    "sharpness",
    "spearman_correlation",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "calibration_curve",
    "reliability_diagram",
    "uncertainty_vs_error_scatter",
]

__version__ = "0.1.0"
