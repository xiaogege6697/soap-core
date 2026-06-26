import pytest
from soap.apps.training.real_adapter import normalize_training_log, describe_fields


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def test_standard_fields(tmp_path):
    csv_path = tmp_path / "standard.csv"
    header = ["step", "loss", "val_loss", "grad_norm", "learning_rate"]
    rows = [
        [1, 0.5, 0.6, 0.01, 0.001],
        [2, 0.45, 0.55, 0.009, 0.0009],
        [3, 0.4, 0.5, 0.008, 0.0008],
        [4, 0.35, 0.45, 0.007, 0.0007],
        [5, 0.3, 0.4, 0.006, 0.0006],
    ]
    _write_csv(csv_path, header, rows)

    result = normalize_training_log(str(csv_path), field_map=None)
    assert hasattr(result, "values")
    assert len(result.values) == 5


def test_wandb_field_map(tmp_path):
    csv_path = tmp_path / "wandb.csv"
    header = ["step", "train/loss", "eval/loss", "train/grad_norm", "lr"]
    rows = [
        [1, 0.5, 0.6, 0.01, 0.001],
        [2, 0.45, 0.55, 0.009, 0.0009],
        [3, 0.4, 0.5, 0.008, 0.0008],
        [4, 0.35, 0.45, 0.007, 0.0007],
        [5, 0.3, 0.4, 0.006, 0.0006],
    ]
    _write_csv(csv_path, header, rows)

    field_map = {
        "train/loss": "loss",
        "eval/loss": "val_loss",
        "train/grad_norm": "grad_norm",
        "lr": "learning_rate",
        "step": "step",
    }
    result = normalize_training_log(str(csv_path), field_map=field_map)
    assert hasattr(result, "values")


def test_negative_grad_norm_raises(tmp_path):
    csv_path = tmp_path / "negative.csv"
    header = ["step", "loss", "val_loss", "grad_norm", "learning_rate"]
    rows = [
        [1, 0.5, 0.6, 0.01, 0.001],
        [2, 0.45, 0.55, -0.1, 0.0009],
        [3, 0.4, 0.5, 0.008, 0.0008],
    ]
    _write_csv(csv_path, header, rows)

    with pytest.raises(ValueError):
        normalize_training_log(str(csv_path), field_map=None)


def test_missing_val_loss_ok(tmp_path):
    csv_path = tmp_path / "no_val_loss.csv"
    header = ["step", "loss", "grad_norm"]
    rows = [
        [1, 0.5, 0.01],
        [2, 0.45, 0.009],
        [3, 0.4, 0.008],
        [4, 0.35, 0.007],
        [5, 0.3, 0.006],
    ]
    _write_csv(csv_path, header, rows)

    result = normalize_training_log(str(csv_path), field_map=None)
    assert hasattr(result, "values")


def test_describe_fields(tmp_path):
    csv_path = tmp_path / "describe.csv"
    header = ["step", "loss", "val_loss", "grad_norm", "learning_rate"]
    rows = [
        [1, 0.5, 0.6, 0.01, 0.001],
        [2, 0.45, 0.55, 0.009, 0.0009],
        [3, 0.4, 0.5, 0.008, 0.0008],
        [4, 0.35, 0.45, 0.007, 0.0007],
        [5, 0.3, 0.4, 0.006, 0.0006],
    ]
    _write_csv(csv_path, header, rows)

    result = describe_fields(str(csv_path))
    assert isinstance(result, dict)
    assert "standard_fields_present" in result
    assert "rows" in result
    assert result["rows"] == 5
