"""Takens delay embedding utilities."""


def takens_embedding(
    series: list[list[float]], delay: int, dimension: int
) -> list[list[float]]:
    """Build delay-coordinate vectors from a univariate or multivariate series."""
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")

    required_length = (dimension - 1) * delay + 1
    if len(series) < required_length:
        raise ValueError(
            "not enough data for Takens embedding: "
            f"need at least {required_length} rows for delay={delay}, "
            f"dimension={dimension}, got {len(series)}"
        )

    embedded: list[list[float]] = []
    output_length = len(series) - (dimension - 1) * delay
    for start_index in range(output_length):
        row: list[float] = []
        for delay_index in range(dimension):
            row.extend(series[start_index + delay_index * delay])
        embedded.append(row)

    return embedded
