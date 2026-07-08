# Changelog

## 0.7.7 - 2026-07-09

### Fixed

- Keep `soap.prediction.smap` as a lazy optional import so the default CLI path works without `numpy`.
- Preserve the zero-dependency `--predictor none` path promised by the README.

### Packaging

- Align package metadata and `soap.__version__` with the v0.7 line.
- Add `readme = "README.md"` and the `soap` console script entry point.

## 0.7.6 - 2026-06-30

### Added

- Class-conditional representation geometry benchmark and metrics.
- End-to-end smoke coverage for `hf_classcond_experiment.py`.
- Documentation for the controlled rank-1 geometry benchmark and its limits.

### Boundary

- Class-conditional geometry remains a controlled benchmark signal, not a default natural-collapse detector.

