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


def test_write_normalized_csv_step_is_string(tmp_path):
    """write_normalized_training_csv 输出首列 step 必须是字符串（step_0/step_1），
    防 SOAP load_csv 把数值 step 当维度（v0.6.6 schema 陷阱、v0.6.7 固化）。"""
    from soap.apps.training.real_adapter import write_normalized_training_csv
    from soap.data.loader import load_csv
    import csv as _csv

    src = tmp_path / "src.csv"
    _write_csv(
        src,
        ["step", "loss", "val_loss", "grad_norm", "learning_rate"],
        [[0, 0.5, 0.6, 0.01, 0.001], [1, 0.45, 0.55, 0.009, 0.0009]],
    )
    out = tmp_path / "out.csv"
    write_normalized_training_csv(str(src), str(out))

    with open(out) as f:
        r = list(_csv.reader(f))
    # 第一列 step 为字符串，非纯数字
    assert r[1][0] == "step_0"
    assert r[2][0] == "step_1"

    # load_csv 读输出 CSV：step 字符串首列被当 timestamp 跳过，剩 4 数值列
    ts = load_csv(str(out))
    assert len(ts.values[0]) == 4
