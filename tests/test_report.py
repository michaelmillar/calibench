from __future__ import annotations

import json

import matplotlib
import matplotlib.figure
import numpy as np

matplotlib.use("Agg")

from calibench.report import Report, audit


def _make_calibrated(n: int = 10_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_pred = rng.uniform(0, 10, size=n)
    y_std = rng.uniform(0.5, 2.0, size=n)
    y_true = y_pred + rng.normal(0.0, y_std)
    return y_true, y_pred, y_std


def _make_overconfident(
    n: int = 10_000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_pred = rng.uniform(0, 10, size=n)
    true_std = rng.uniform(1.0, 3.0, size=n)
    y_std = true_std * 0.3
    y_true = y_pred + rng.normal(0.0, true_std)
    return y_true, y_pred, y_std


class TestAudit:
    def test_returns_report(self) -> None:
        y_true, y_pred, y_std = _make_calibrated()
        result = audit(y_true, y_pred, y_std)
        assert isinstance(result, Report)

    def test_calibrated_verdict(self) -> None:
        y_true, y_pred, y_std = _make_calibrated()
        result = audit(y_true, y_pred, y_std)
        assert result.verdict == "well-calibrated"

    def test_overconfident_verdict(self) -> None:
        y_true, y_pred, y_std = _make_overconfident()
        result = audit(y_true, y_pred, y_std)
        assert result.verdict == "overconfident"

    def test_recalibration_improves_ece(self) -> None:
        y_true, y_pred, y_std = _make_overconfident()
        result = audit(y_true, y_pred, y_std)
        assert result.recalibrated_ece < result.ece

    def test_residuals_computed_correctly(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=100)
        result = audit(y_true, y_pred, y_std)
        expected = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
        np.testing.assert_array_almost_equal(result.residuals, expected)

    def test_temperature_matches_calibrator(self) -> None:
        y_true, y_pred, y_std = _make_overconfident()
        result = audit(y_true, y_pred, y_std)
        assert result.temperature == result.calibrator.temperature


class TestReportToDict:
    def test_returns_dict(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_dict()
        assert isinstance(result, dict)

    def test_json_serialisable(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_dict()
        json.dumps(result)

    def test_excludes_arrays(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_dict()
        for val in result.values():
            assert not isinstance(val, np.ndarray)

    def test_contains_all_metrics(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_dict()
        expected_keys = {
            "ece",
            "miscalibration_area",
            "sharpness",
            "coverage_at_90",
            "spearman_correlation",
            "recalibrated_ece",
            "recalibrated_miscalibration_area",
            "recalibrated_sharpness",
            "recalibrated_coverage_at_90",
            "recalibrated_spearman_correlation",
            "verdict",
            "temperature",
        }
        assert expected_keys == set(result.keys())


class TestReportStr:
    def test_returns_string(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = str(audit(y_true, y_pred, y_std))
        assert isinstance(result, str)

    def test_contains_verdict(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        report = audit(y_true, y_pred, y_std)
        assert report.verdict in str(report)


class TestReportToMarkdown:
    def test_returns_string(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_markdown()
        assert isinstance(result, str)

    def test_contains_table_header(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_markdown()
        assert "| Metric |" in result

    def test_contains_verdict(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        result = audit(y_true, y_pred, y_std).to_markdown()
        assert "**Verdict:**" in result


class TestReportPlot:
    def test_returns_figure(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        fig = audit(y_true, y_pred, y_std).plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_has_three_axes(self) -> None:
        y_true, y_pred, y_std = _make_calibrated(n=500)
        fig = audit(y_true, y_pred, y_std).plot()
        assert len(fig.axes) == 3
        import matplotlib.pyplot as plt

        plt.close("all")
