
# core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import SimpleITK as sitk
import argparse
from label_analysis.geometry import LabelMapGeometry
from utilz.fileio import load_json
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import SimpleITK as sitk


# -------------------------------
# Data structures
# -------------------------------

# -------------------------------
# Markups parsing (Slicer .mrk.json)
# -------------------------------

@dataclass
class BBox:
    imin: int
    isize: int
    jmin: int
    jsize: int
    kmin: int
    ksize: int

    def as_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (self.imin, self.isize, self.jmin, self.jsize, self.kmin, self.ksize)

    def as_slices(self) -> Tuple[slice, slice, slice]:
        return (
            slice(self.imin, self.imin + self.isize),
            slice(self.jmin, self.jmin + self.jsize),
            slice(self.kmin, self.kmin + self.ksize),
        )

    def start(self) -> Tuple[int, int, int]:
        return (self.imin, self.jmin, self.kmin)

    def end_exclusive(self) -> Tuple[int, int, int]:
        return (self.imin + self.isize, self.jmin + self.jsize, self.kmin + self.ksize)

    def end_inclusive(self) -> Tuple[int, int, int]:
        ei, ej, ek = self.end_exclusive()
        return (ei - 1, ej - 1, ek - 1)

    def to_start_size(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.start(), (self.isize, self.jsize, self.ksize)

    def contains(self, other: "BBox") -> bool:
        a0_i, a0_j, a0_k = self.start()
        a1_i, a1_j, a1_k = self.end_exclusive()
        b0_i, b0_j, b0_k = other.start()
        b1_i, b1_j, b1_k = other.end_exclusive()
        return (
            (a0_i <= b0_i <= b1_i <= a1_i)
            and (a0_j <= b0_j <= b1_j <= a1_j)
            and (a0_k <= b0_k <= b1_k <= a1_k)
        )

    def intersects(self, other: Union["BBox", Tuple[int, int, int]]) -> bool:
        """Check intersection.

        - If `other` is a BBox: return True if the boxes overlap.
        - If `other` is a 3-tuple (i,j,k): return True if the point lies inside this box.
        """
        a0_i, a0_j, a0_k = self.start()
        a1_i, a1_j, a1_k = self.end_exclusive()

        if isinstance(other, BBox):
            b0_i, b0_j, b0_k = other.start()
            b1_i, b1_j, b1_k = other.end_exclusive()
            return (
                (a0_i < b1_i and b0_i < a1_i)
                and (a0_j < b1_j and b0_j < a1_j)
                and (a0_k < b1_k and b0_k < a1_k)
            )
        else:
            i, j, k = other
            return (a0_i <= i < a1_i and a0_j <= j < a1_j and a0_k <= k < a1_k)

    def union(self, other: "BBox") -> "BBox":
        a0_i, a0_j, a0_k = self.start()
        a1_i, a1_j, a1_k = self.end_exclusive()
        b0_i, b0_j, b0_k = other.start()
        b1_i, b1_j, b1_k = other.end_exclusive()
        u0_i = min(a0_i, b0_i)
        u0_j = min(a0_j, b0_j)
        u0_k = min(a0_k, b0_k)
        u1_i = max(a1_i, b1_i)
        u1_j = max(a1_j, b1_j)
        u1_k = max(a1_k, b1_k)
        return BBox(
            imin=u0_i, isize=u1_i - u0_i,
            jmin=u0_j, jsize=u1_j - u0_j,
            kmin=u0_k, ksize=u1_k - u0_k,
        )

    @staticmethod
    def merge(bboxes: Iterable["BBox"]) -> Optional["BBox"]:
        items = list(bboxes)
        if not items:
            return None
        out = items[0]
        for b in items[1:]:
            out = out.union(b)
        return out




def points_lps_mm_to_indices(points_lps_mm: np.ndarray, img: sitk.Image) -> np.ndarray:
    if points_lps_mm.ndim != 2 or points_lps_mm.shape[1] != 3:
        raise ValueError("points_lps_mm must be (N,3)")
    idxs = np.zeros_like(points_lps_mm, dtype=np.float64)
    for n, pt in enumerate(points_lps_mm):
        i, j, k = img.TransformPhysicalPointToContinuousIndex(tuple(map(float, pt)))
        idxs[n, 0] = i
        idxs[n, 1] = j
        idxs[n, 2] = k
    return idxs


def bbox_from_indices(idx_float: np.ndarray, size_ijk: Tuple[int, int, int], pad: int = 0) -> BBox:
    if idx_float.ndim != 2 or idx_float.shape[1] != 3:
        raise ValueError("idx_float must be (N,3)")

    mins = np.floor(idx_float.min(axis=0)).astype(int)
    maxs = np.ceil(idx_float.max(axis=0)).astype(int) - 1

    mins -= pad
    maxs += pad

    ni, nj, nk = size_ijk
    mins[0] = max(0, mins[0]); maxs[0] = min(ni - 1, maxs[0])
    mins[1] = max(0, mins[1]); maxs[1] = min(nj - 1, maxs[1])
    mins[2] = max(0, mins[2]); maxs[2] = min(nk - 1, maxs[2])

    if maxs[0] < mins[0]:
        maxs[0] = min(mins[0] + 1, ni - 1)
    if maxs[1] < mins[1]:
        maxs[1] = min(mins[1] + 1, nj - 1)
    if maxs[2] < mins[2]:
        maxs[2] = min(mins[2] + 1, nk - 1)

    isize = maxs[0] - mins[0] + 1
    jsize = maxs[1] - mins[1] + 1
    ksize = maxs[2] - mins[2] + 1

    return BBox(
        imin=int(mins[0]), isize=isize,
        jmin=int(mins[1]), jsize=jsize,
        kmin=int(mins[2]), ksize=ksize,
    )


def load_markups_points_lps(mrk_path: Path) -> np.ndarray:
    payload: Dict[str, Any] = json.loads(Path(mrk_path).read_text())
    cs = str(payload.get("coordinateSystem","LPS")).upper()
    pts = []
    for m in payload["markups"]:
        for cp in m.get("controlPoints", []):
            pos = cp["position"]
            pts.append([float(pos[0]), float(pos[1]), float(pos[2])])
    P = np.asarray(pts, dtype=float)
    if cs == "RAS":
        P[:, :2] *= -1
    return P

def load_markups_points_indices(mrk_path: Path, img: [sitk.Image,str,Path]) -> np.ndarray:
    if not isinstance(img, sitk.Image):
        img = sitk.ReadImage(str(img))
    pts = load_markups_points_lps(mrk_path)
    return points_lps_mm_to_indices(pts, img)


def bbox_from_markup_file(mrk_path: Path, image_path: Path, pad: int = 0) -> BBox:
    img = sitk.ReadImage(str(image_path))
    pts = load_markups_points_lps(mrk_path)
    idx = points_lps_mm_to_indices(pts, img)
    return bbox_from_indices(idx, img.GetSize(), pad=pad)
