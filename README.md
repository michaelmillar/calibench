<p align="center">
  <img src="assets/calibench.svg" width="80" height="80"/>
</p>
<h1 align="center">calibench</h1>
<p align="center">One-line uncertainty audit for ML models.</p>

## Quickstart

```bash
pip install calibench
```

```python
import numpy as np
from calibench import audit

y_true = np.array([1.0, 2.5, 3.1, 4.8])
y_pred = np.array([1.1, 2.3, 3.4, 4.6])
y_std = np.array([0.5, 1.0, 0.8, 1.2])

report = audit(y_true, y_pred, y_std)
print(report)
```

```
calibench audit
===============
Verdict: well-calibrated (temperature = 1.03)

                      Before       After
ECE                   0.0320      0.0110
Miscalibration area   0.0180      0.0060
Sharpness             2.4100      2.5500
Coverage @ 90%        0.8900      0.9100
Spearman r            0.7100      0.7300
```

## What it does

Pass predictions, uncertainties, and ground truth. Get back a structured report that tells you whether your uncertainty estimates are trustworthy, where they fail, and how to fix them.

`audit()` runs five calibration metrics before and after automatic recalibration via temperature scaling, then returns a `Report` with a verdict, all numbers, and ready-made plots.

```python
report.plot()

report.to_dict()

report.to_markdown()
```

The `Report` includes a fitted `TemperatureScaler` you can apply to new predictions:

```python
calibrated_std = report.calibrator.transform(new_std)
```

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
