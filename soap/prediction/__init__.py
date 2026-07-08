"""State transition and probabilistic forecasting."""

from soap.prediction.simplex import prediction_skill, simplex_predict

__all__ = ["prediction_skill", "simplex_predict", "smap_predict", "smap_skill"]


def __getattr__(name: str):
    if name in {"smap_predict", "smap_skill"}:
        from soap.prediction.smap import smap_predict, smap_skill

        return {"smap_predict": smap_predict, "smap_skill": smap_skill}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
