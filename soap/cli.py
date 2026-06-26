from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path

from soap.attractor.clustering import kmeans
from soap.data.loader import load_csv
from soap.data.preprocessing import sliding_windows, standardize
from soap.dimension.pca_selector import project_to_dimension_with_pca, score_dimensions_with_pca
from soap.dimension.selector import project_to_dimension, score_dimensions, select_optimal_dimension
from soap.embedding import takens_embedding
from soap.metrics import estimate_delay_ami, estimate_embedding_dimension_fnn
from soap.prediction.transition import next_state_probabilities, transition_matrix
from soap.report.generator import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="SOAP-Core CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    example_parser = subparsers.add_parser("generate-example", help="Generate synthetic cyclic data.")
    example_parser.add_argument("--output", required=True)
    example_parser.add_argument("--length", type=int, default=300)

    lorenz_parser = subparsers.add_parser("generate-lorenz", help="Generate synthetic Lorenz attractor data.")
    lorenz_parser.add_argument("--output", required=True)
    lorenz_parser.add_argument("--length", type=int, default=2000)
    lorenz_parser.add_argument("--dt", type=float, default=0.01)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze CSV time series.")
    analyze_parser.add_argument("csv_path")
    analyze_parser.add_argument("--window-size", type=int, default=1)
    analyze_parser.add_argument("--max-dim", type=int, default=8)
    analyze_parser.add_argument("--clusters", type=int, default=3)
    analyze_parser.add_argument("--method", choices=["variance", "pca"], default="variance")
    analyze_parser.add_argument("--embedding", choices=["window", "takens"], default="window")
    analyze_parser.add_argument("--delay", default="auto")
    analyze_parser.add_argument("--embedding-dim", default="auto")
    analyze_parser.add_argument("--max-delay", type=int, default=50)
    analyze_parser.add_argument("--max-embedding-dim", type=int, default=8)
    analyze_parser.add_argument("--predictor", choices=["none", "simplex", "smap"], default="none")
    analyze_parser.add_argument("--recurrence", action="store_true")
    analyze_parser.add_argument("--recurrence-radius", type=float, default=None)
    analyze_parser.add_argument("--recurrence-quantile", type=float, default=0.1)
    analyze_parser.add_argument("--smap-theta", type=float, default=2.0)
    analyze_parser.add_argument("--neighbors", type=int, default=10)
    analyze_parser.add_argument("--output-dir", default="outputs")
    analyze_parser.add_argument("--report-dir", default="reports")

    args = parser.parse_args()
    if args.command == "generate-example":
        generate_example(args.output, args.length)
    elif args.command == "generate-lorenz":
        generate_lorenz(args.output, args.length, args.dt)
    elif args.command == "analyze":
        analyze(args)


def generate_example(output: str, length: int) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "x1", "x2", "x3", "x4", "x5"])
        for index in range(length):
            phase = index / 12.0
            row = [
                f"t_{index}",
                math.sin(phase),
                math.sin(phase + 1.25),
                math.sin(phase + 2.50),
                math.sin(phase + 3.75),
                math.sin(phase + 5.00),
            ]
            writer.writerow(row)
    print(f"Example written: {path}")


def generate_lorenz(output: str, length: int, dt: float) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = 1.0, 1.0, 1.0

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "x", "y", "z"])
        for index in range(length):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            writer.writerow([f"t_{index}", x, y, z])
    print(f"Lorenz example written: {path}")


def analyze(args: argparse.Namespace) -> None:
    if args.max_dim < 1:
        raise ValueError("max_dim 必须 >= 1。")
    if args.clusters < 1:
        raise ValueError("clusters 必须 >= 1。")
    if args.max_delay < 1:
        raise ValueError("max_delay 必须 >= 1。")
    if args.max_embedding_dim < 1:
        raise ValueError("max_embedding_dim 必须 >= 1。")
    if args.neighbors < 1:
        raise ValueError("neighbors 必须 >= 1。")
    if args.recurrence_radius is not None and args.recurrence_radius <= 0:
        raise ValueError("recurrence_radius 必须 > 0。")
    if not 0 < args.recurrence_quantile <= 1:
        raise ValueError("recurrence_quantile 必须在 (0, 1] 内。")
    if args.smap_theta < 0:
        raise ValueError("smap_theta 必须 >= 0。")

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    series = load_csv(args.csv_path)
    scaled = standardize(series.values)
    states, embedding_metadata = _build_states(scaled, args)

    if args.method == "pca":
        scores = score_dimensions_with_pca(states, args.max_dim)
    else:
        scores = score_dimensions(states, args.max_dim)
    optimal = select_optimal_dimension(scores)
    if args.method == "pca":
        embedding = project_to_dimension_with_pca(states, optimal.dimension)
    else:
        embedding = project_to_dimension(states, optimal.dimension)

    labels, centroids = kmeans(embedding, args.clusters)
    matrix = transition_matrix(labels, args.clusters)
    probabilities = next_state_probabilities(labels, matrix)
    prediction_diagnostics = _run_prediction_diagnostics(embedding, args)
    recurrence_diagnostics = _run_recurrence_diagnostics(embedding, args)

    _write_scores(output_dir / "dimension_scores.csv", scores)
    _write_embedding(output_dir / "embedding.csv", embedding, labels)
    _write_matrix(output_dir / "transition_matrix.csv", matrix)
    _write_score_svg(output_dir / "dimension_scores.svg", scores, optimal.dimension)
    _write_embedding_svg(output_dir / "embedding.svg", embedding, labels)
    write_report(
        report_dir / "report.md",
        optimal,
        scores,
        probabilities,
        method=args.method,
        embedding_metadata=embedding_metadata,
        prediction_diagnostics=prediction_diagnostics,
        recurrence_diagnostics=recurrence_diagnostics,
    )

    print(f"Optimal effective dimension: {optimal.dimension}")
    print(f"Dimension method: {args.method}")
    print(f"Embedding method: {embedding_metadata['embedding_method']}")
    if args.embedding == "takens":
        print(f"Takens delay: {embedding_metadata['takens_delay']}")
        print(f"Takens embedding dimension: {embedding_metadata['takens_embedding_dimension']}")
    if prediction_diagnostics.get("status") == "unavailable":
        print(f"Prediction diagnostics skipped: {prediction_diagnostics['reason']}")
    if recurrence_diagnostics.get("status") == "unavailable":
        print(f"Recurrence diagnostics skipped: {recurrence_diagnostics['reason']}")
    print(f"Report: {report_dir / 'report.md'}")


def _run_prediction_diagnostics(embedding: list[list[float]], args: argparse.Namespace) -> dict[str, object]:
    if args.predictor == "none":
        return {"status": "disabled", "predictor": "none"}

    module_name = f"soap.prediction.{args.predictor}"
    try:
        module = importlib.import_module(module_name)
        if args.predictor == "simplex" and callable(getattr(module, "simplex_predict", None)):
            predicted = module.simplex_predict(embedding, neighbors=args.neighbors)
            result = module.prediction_skill(predicted, embedding[1:])
        elif args.predictor == "smap" and callable(getattr(module, "smap_predict", None)):
            predicted = module.smap_predict(embedding, neighbors=args.neighbors, theta=args.smap_theta)
            result = module.smap_skill(predicted, embedding[1:])
        else:
            function = _first_available(
                module,
                [
                    f"evaluate_{args.predictor}",
                    f"{args.predictor}_diagnostics",
                    "prediction_diagnostics",
                    "evaluate",
                ],
            )
            if args.predictor == "smap":
                result = function(embedding, neighbors=args.neighbors, theta=args.smap_theta)
            else:
                result = function(embedding, neighbors=args.neighbors)
    except Exception as error:
        return {
            "status": "unavailable",
            "predictor": args.predictor,
            "reason": f"{module_name} 不可用或 API 尚未接入：{error}",
        }

    if isinstance(result, dict):
        return {"status": "ok", "predictor": args.predictor, **result}
    return {"status": "ok", "predictor": args.predictor, "result": result}


def _run_recurrence_diagnostics(embedding: list[list[float]], args: argparse.Namespace) -> dict[str, object]:
    if not args.recurrence:
        return {"status": "disabled", "enabled": False}

    module_name = "soap.metrics.recurrence"
    try:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            module_name = "soap.recurrence.diagnostics"
            module = importlib.import_module(module_name)

        if callable(getattr(module, "recurrence_matrix", None)) and callable(getattr(module, "recurrence_summary", None)):
            matrix = module.recurrence_matrix(
                embedding,
                radius=args.recurrence_radius,
                quantile=args.recurrence_quantile,
            )
            result = {
                **module.recurrence_summary(matrix),
                "matrix_size": len(matrix),
                "radius": args.recurrence_radius if args.recurrence_radius is not None else "auto",
                "quantile": args.recurrence_quantile,
            }
        else:
            function = _first_available(
                module,
                [
                    "compute_recurrence_diagnostics",
                    "recurrence_diagnostics",
                    "evaluate",
                ],
            )
            result = function(
                embedding,
                radius=args.recurrence_radius,
                quantile=args.recurrence_quantile,
            )
    except Exception as error:
        return {
            "status": "unavailable",
            "enabled": True,
            "reason": f"{module_name} 不可用或 API 尚未接入：{error}",
        }

    if isinstance(result, dict):
        return {"status": "ok", "enabled": True, **result}
    return {"status": "ok", "enabled": True, "result": result}


def _first_available(module, names: list[str]):
    for name in names:
        function = getattr(module, name, None)
        if callable(function):
            return function
    raise AttributeError(f"缺少候选函数：{', '.join(names)}")


def _build_states(values: list[list[float]], args: argparse.Namespace) -> tuple[list[list[float]], dict[str, str | int]]:
    if args.embedding == "window":
        states = sliding_windows(values, args.window_size)
        return states, {
            "embedding_method": "window",
            "window_size": args.window_size,
            "effective_sample_count": len(states),
        }

    max_delay = min(args.max_delay, len(values) - 1)
    if max_delay < 1:
        raise ValueError("Takens embedding 至少需要 2 行数据。")

    delay_source = "manual"
    if args.delay == "auto":
        delay = estimate_delay_ami(values, max_delay=max_delay)
        delay_source = "ami"
    else:
        delay = _parse_positive_int(args.delay, "delay")

    dimension_source = "manual"
    if args.embedding_dim == "auto":
        embedding_dimension = estimate_embedding_dimension_fnn(
            values,
            delay=delay,
            max_dim=args.max_embedding_dim,
        )
        dimension_source = "fnn"
    else:
        embedding_dimension = _parse_positive_int(args.embedding_dim, "embedding_dim")

    states = takens_embedding(values, delay=delay, dimension=embedding_dimension)
    return states, {
        "embedding_method": "takens",
        "takens_delay": delay,
        "delay_source": delay_source,
        "takens_embedding_dimension": embedding_dimension,
        "dimension_source": dimension_source,
        "ami_max_delay": max_delay,
        "max_embedding_dim": args.max_embedding_dim,
        "effective_sample_count": len(states),
    }


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} 必须是 auto 或正整数。") from error
    if parsed < 1:
        raise ValueError(f"{name} 必须 >= 1。")
    return parsed


def _write_scores(path: Path, scores) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dimension", "reconstruction_error", "prediction_error", "complexity_penalty", "total_score"])
        for score in scores:
            writer.writerow([
                score.dimension,
                score.reconstruction_error,
                score.prediction_error,
                score.complexity_penalty,
                score.total_score,
            ])


def _write_embedding(path: Path, embedding: list[list[float]], labels: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        width = len(embedding[0])
        writer.writerow([f"dim_{index + 1}" for index in range(width)] + ["attractor_label"])
        for row, label in zip(embedding, labels):
            writer.writerow(row + [label])


def _write_matrix(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"A{index}" for index in range(len(matrix))])
        writer.writerows(matrix)


def _write_score_svg(path: Path, scores, optimal_dimension: int) -> None:
    width, height = 720, 420
    margin = 56
    max_score = max(score.total_score for score in scores) or 1.0
    min_score = min(score.total_score for score in scores)
    span = max_score - min_score or 1.0

    points = []
    for index, score in enumerate(scores):
        x = margin + index * ((width - 2 * margin) / max(1, len(scores) - 1))
        y = height - margin - ((score.total_score - min_score) / span) * (height - 2 * margin)
        points.append((x, y, score))

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    circles = []
    for x, y, score in points:
        color = "#ff7a59" if score.dimension == optimal_dimension else "#4f8cff"
        radius = 6 if score.dimension == optimal_dimension else 4
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" />')
        circles.append(f'<text x="{x:.2f}" y="{height - 24}" text-anchor="middle" font-size="12">{score.dimension}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#10141f"/>
  <text x="{width / 2}" y="30" text-anchor="middle" fill="#f2f5ff" font-size="20" font-family="Arial">Dimension Score Curve</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#7d879c"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#7d879c"/>
  <text x="{width / 2}" y="{height - 4}" text-anchor="middle" fill="#bac4d8" font-size="13">dimension d</text>
  <text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" fill="#bac4d8" font-size="13">total score lower is better</text>
  <polyline points="{polyline}" fill="none" stroke="#4f8cff" stroke-width="3"/>
  {''.join(circles)}
  <text x="{width - margin}" y="58" text-anchor="end" fill="#ffb199" font-size="14">d* = {optimal_dimension}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _write_embedding_svg(path: Path, embedding: list[list[float]], labels: list[int]) -> None:
    width, height = 720, 520
    margin = 52
    if not embedding or len(embedding[0]) < 2:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    xs = [row[0] for row in embedding]
    ys = [row[1] for row in embedding]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    palette = ["#4f8cff", "#ff7a59", "#5ee6a8", "#d889ff", "#ffd166", "#70d6ff"]

    dots = []
    for row, label in zip(embedding, labels):
        x = margin + ((row[0] - min_x) / span_x) * (width - 2 * margin)
        y = height - margin - ((row[1] - min_y) / span_y) * (height - 2 * margin)
        color = palette[label % len(palette)]
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.78" />')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#10141f"/>
  <text x="{width / 2}" y="30" text-anchor="middle" fill="#f2f5ff" font-size="20" font-family="Arial">Low-Dimensional State Space</text>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#7d879c"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#7d879c"/>
  <text x="{width / 2}" y="{height - 8}" text-anchor="middle" fill="#bac4d8" font-size="13">dimension 1</text>
  <text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" fill="#bac4d8" font-size="13">dimension 2</text>
  {''.join(dots)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
