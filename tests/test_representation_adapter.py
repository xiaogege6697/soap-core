import pytest
from pathlib import Path
from soap.apps.training.representation_adapter import (
    summarize_representation_metrics,
    detect_representation_collapse,
    METRIC_COLS,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_normal_not_collapse():
    s = summarize_representation_metrics(EXAMPLES / "torch_training_normal_repr_enhanced.csv")
    d = detect_representation_collapse(s)
    assert d["is_collapse"] is False
    assert d["status"] == "ok"


def test_mode_collapse_detected():
    s = summarize_representation_metrics(EXAMPLES / "torch_training_mode_collapse_repr_enhanced.csv")
    d = detect_representation_collapse(s)
    assert d["is_collapse"] is True
    assert d["status"] == "collapse"
    assert d["collapse_score_final"] > 0.9
    assert d["effective_rank_final"] < 1.5


def test_overfit_not_collapse():
    s = summarize_representation_metrics(EXAMPLES / "torch_training_overfit_repr_enhanced.csv")
    d = detect_representation_collapse(s)
    assert d["is_collapse"] is False
    assert d["status"] == "ok"


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,loss\nstep_0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        summarize_representation_metrics(p)


def test_summary_has_all_fields():
    s = summarize_representation_metrics(EXAMPLES / "torch_training_normal_repr_enhanced.csv")
    for col in METRIC_COLS:
        assert col in s
        for key in ["start", "end", "mean", "min", "max", "drop_ratio"]:
            assert key in s[col]
