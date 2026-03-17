from .totalseg import TotalSegmenterLabels

__all__ = ["TotalSegmenterLabels", "LabelMapGeometryPT"]


def __getattr__(name):
    if name == "LabelMapGeometryPT":
        from .geometry_pt import LabelMapGeometryPT

        return LabelMapGeometryPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
