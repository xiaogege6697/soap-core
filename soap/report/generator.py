from __future__ import annotations

from pathlib import Path

from soap.dimension.selector import DimensionScore


def write_report(
    path: str | Path,
    optimal: DimensionScore,
    scores: list[DimensionScore],
    probabilities: list[float],
    method: str = "variance",
    embedding_metadata: dict[str, str | int] | None = None,
    prediction_diagnostics: dict[str, object] | None = None,
    recurrence_diagnostics: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = embedding_metadata or {"embedding_method": "window"}
    lines = [
        "# SOAP-Core Analysis Report",
        "",
        f"Dimension method: **{method}**",
        "",
        f"Embedding method: **{metadata['embedding_method']}**",
        "",
        f"Optimal effective dimension: **{optimal.dimension}**",
        "",
        "## Dimension Scores",
        "",
        "| d | reconstruction_error | prediction_error | complexity_penalty | total_score |",
        "|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            f"| {score.dimension} | {score.reconstruction_error:.6f} | "
            f"{score.prediction_error:.6f} | {score.complexity_penalty:.6f} | "
            f"{score.total_score:.6f} |"
        )

    lines.extend(["", "## Embedding Metadata", ""])
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Next-State Probabilities", ""])
    for index, probability in enumerate(probabilities):
        lines.append(f"- Attractor A{index}: {probability:.3f}")

    lines.extend(["", "## Prediction Diagnostics", ""])
    _append_diagnostics(lines, prediction_diagnostics or {"status": "disabled", "predictor": "none"})

    lines.extend(["", "## Recurrence Diagnostics", ""])
    _append_diagnostics(lines, recurrence_diagnostics or {"status": "disabled", "enabled": False})

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_diagnostics(lines: list[str], diagnostics: dict[str, object]) -> None:
    for key, value in diagnostics.items():
        lines.append(f"- {key}: {_format_diagnostic_value(value)}")


def _format_diagnostic_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
