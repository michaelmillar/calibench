<p align="center">
  <img src="assets/calibench.svg" width="80" height="80"/>
</p>
<h1 align="center">calibench</h1>
<p align="center">Calibration metrics and diagnostics for uncertainty quantification.</p>

## Quickstart

```bash
pip install calibench
```

```python
import numpy as np
from calibench import miscalibration_area, reliability_diagram

predicted_std = np.array([0.5, 1.0, 1.5, 2.0])
residuals = np.array([0.3, -0.8, 1.2, -1.9])

area = miscalibration_area(predicted_std, residuals)
print(f"Miscalibration area = {area:.4f}")

ax = reliability_diagram(predicted_std, residuals)
```

## Why Calibration Matters

A model that says "I am 90% confident" should be right about 90% of the time. When predicted uncertainties do not match observed error rates, downstream decisions built on those predictions become unreliable. Calibration benchmarking measures this alignment and helps you fix it.

## Metrics

| Function | What it measures |
|---|---|
| `miscalibration_area` | Area between the reliability curve and the ideal diagonal |
| `expected_calibration_error` | Binned ECE across confidence levels |
| `sharpness` | Average width of prediction intervals |
| `coverage_at_confidence` | Fraction of true values falling within a given confidence interval |
| `spearman_correlation` | Rank correlation between predicted uncertainty and absolute error |

## Calibrators

**IsotonicCalibrator** fits an isotonic regression to map raw predicted standard deviations to calibrated ones. It is flexible and nonparametric.

```python
from calibench import IsotonicCalibrator

cal = IsotonicCalibrator()
calibrated_std = cal.fit_transform(predicted_std, residuals)
```

**TemperatureScaler** learns a single scalar T that rescales all uncertainties. It is simple and robust when the model is uniformly miscalibrated.

```python
from calibench import TemperatureScaler

scaler = TemperatureScaler()
calibrated_std = scaler.fit_transform(predicted_std, residuals)
print(f"Learned temperature = {scaler.temperature:.3f}")
```

## Visualisations

**reliability_diagram** plots expected vs observed coverage at multiple quantile levels.

**calibration_curve** plots mean predicted standard deviation against mean absolute residual in bins.

**uncertainty_vs_error_scatter** produces a scatter plot of predicted uncertainty against absolute error, with a y=x reference line.

All plotting functions return a matplotlib `Axes` object and accept an optional `ax` parameter to draw on an existing figure.
