#!/usr/bin/env python3

"""
Compute axis-aligned bounding boxes (in voxel index space) from Slicer ClosedCurve
markup files (.mrk.json), using a single reference image.

Modern schema only:
- Expects top-level "coordinateSystem" of "LPS" or "RAS".
- Expects payload["markups"][*]["controlPoints"][*]["position"] as a 3-array or {x,y,z}.

Outputs per-file integer bboxes and an optional merged bbox.
Instead of max indices, stores sizes along each axis.
"""
from __future__ import annotations

import argparse

from utilz.helpers import set_autoreload

from label_analysis.helpers import get_labels, to_cc
from label_analysis.geometry import LabelMapGeometry
from utilz.fileio import load_json
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import SimpleITK as sitk

from label_analysis.helpers import remap_single_label


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

# The rest of the script (markups parsing, bbox_from_indices, CLI, etc.) remains unchanged.
def _detect_coord_system(payload: Dict[str, Any]) -> str:
    """Return 'LPS' or 'RAS' from the payload (modern schema only)."""
    cs = str(payload.get("coordinateSystem", "LPS")).upper()
    if cs not in ("LPS", "RAS"):
        raise ValueError("Unsupported coordinateSystem. Expected 'LPS' or 'RAS'.")
    return cs


def extract_points_mm(payload: Dict[str, Any]) -> np.ndarray:
    """Extract control point positions (mm) from modern Slicer markups JSON."""
    mlist = payload.get("markups")
    if not isinstance(mlist, list):
        raise ValueError("Missing 'markups' list in markup payload.")

    pts: List[List[float]] = []
    for m in mlist:
        cpoints = m.get("controlPoints") or []
        for cp in cpoints:
            if "position" not in cp:
                continue
            pos = cp["position"]
            if isinstance(pos, dict):
                pts.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
            else:
                pts.append([float(pos[0]), float(pos[1]), float(pos[2])])

    if not pts:
        raise ValueError("No control points with 'position' found in markups.")

    return np.asarray(pts, dtype=np.float64)


def _ras_to_lps(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


def load_markups_points_lps(mrk_path: Path) -> np.ndarray:
    with open(mrk_path, "r") as f:
        payload = json.load(f)
    pts = extract_points_mm(payload)
    cs = _detect_coord_system(payload)
    if cs == "RAS":
        pts = _ras_to_lps(pts)
    return pts


# -------------------------------
# Physical (mm, LPS) -> voxel index (i,j,k)
# -------------------------------

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


# -------------------------------
# Bounding box construction in index space
# -------------------------------

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


# -------------------------------
# High-level API
# -------------------------------

def bbox_from_markup_file(mrk_path: Path, image_path: Path, pad: int = 0) -> BBox:
    img = sitk.ReadImage(str(image_path))
    pts_lps = load_markups_points_lps(Path(mrk_path))
    idx = points_lps_mm_to_indices(pts_lps, img)
    return bbox_from_indices(idx, img.GetSize(), pad=pad)


def bboxes_from_markup_files(mrk_paths: Iterable[Path], image_path: Path, pad: int = 0) -> Dict[Path, BBox]:
    img = sitk.ReadImage(str(image_path))
    size = img.GetSize()
    out: Dict[Path, BBox] = {}
    for p in mrk_paths:
        pts_lps = load_markups_points_lps(Path(p))
        idx = points_lps_mm_to_indices(pts_lps, img)
        out[Path(p)] = bbox_from_indices(idx, size, pad=pad)
    return out


# -------------------------------
# CLI
# -------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute bbox(es) from Slicer .mrk.json ClosedCurve(s)")
    ap.add_argument("--image", required=True, type=Path, help="Reference image (NIfTI/MHA/etc.)")
    ap.add_argument("--markups", required=True, nargs="+", type=Path, help="One or more .mrk.json files")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to write per-file bbox CSV")
    ap.add_argument("--pad", type=int, default=0, help="Padding in voxels added to all sides")
    return ap.parse_args()


def _write_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "file", "imin", "isize", "jmin", "jsize", "kmin", "ksize",
        "slice_i", "slice_j", "slice_k",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = _parse_args()
    img = sitk.ReadImage(str(args.image))

    results: Dict[Path, BBox] = {}
    rows: List[Dict[str, Any]] = []
    for p in args.markups:
        pts = load_markups_points_lps(p)
        idx = points_lps_mm_to_indices(pts, img)
        bb = bbox_from_indices(idx, img.GetSize(), pad=args.pad)
        results[p] = bb
        si, sj, sk = bb.as_slices()
        rows.append({
            "file": str(p),
            "imin": bb.imin, "isize": bb.isize,
            "jmin": bb.jmin, "jsize": bb.jsize,
            "kmin": bb.kmin, "ksize": bb.ksize,
            "slice_i": f"{si.start}:{si.stop}",
            "slice_j": f"{sj.start}:{sj.stop}",
            "slice_k": f"{sk.start}:{sk.stop}",
        })

    for p, bb in results.items():
        print(f"{p.name}: bbox (imin,isize,jmin,jsize,kmin,ksize) = {bb.as_tuple()}")

    merged = BBox.merge(results.values())
    if merged:
        print(f"MERGED: {merged.as_tuple()}")

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")


# %%
if __name__ == "__main__":
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    set_autoreload()

    lm_fn = "/home/ub/code/label_analysis/label_analysis/files/two_lesions.nrrd"
    lm2_fn = "/home/ub/Documents/nodes_90_411Ta_CAP1p5SoftTissue.nii.gz_2-Segment_1-label.nrrd"
    point2_fn = "/home/ub/Documents/nodes_90.mrk.json"

    cc_fn  = "/home/ub/code/label_analysis/label_analysis/files/two_lesions_cc.mrk.json"
    point_fn = "/home/ub/code/label_analysis/label_analysis/files/two_lesions_point.mrk.json"
    lm = sitk.ReadImage(lm2_fn)
    lmcc = to_cc(lm)
# %%

    img=lmcc
    get_labels(lmcc)
    arr = sitk.GetArrayFromImage(img)
    arr = np.unique(arr)
    arr_int = [int(a) for a in arr if a != 0]
# %%

    cc = load_json(cc_fn)
    point = load_json(point_fn)

# %%
   
    L = LabelMapGeometry(lm2_fn,compute_feret=False)
    L.nbrhoods 
    L.fil.ComputeOrientedBoundingBoxOn()
    L.Execute(L.lm_cc)

    L.lm_org = sitk.Cast(L.lm_org,sitk.sitkUInt32)

    L.create_lm_cc()
    
    
# %%
    point = load_json(point2_fn)
    pts = load_markups_points_lps(cc_fn)
    idx = points_lps_mm_to_indices(pts, lm)
    bb = bbox_from_indices(idx, lm.GetSize(), pad=0)
# %%
# %%
    L.nbrhoods['bbox']
    bb1 = L.nbrhoods['bbox'][0]
    B= BBox(bb1[0],bb1[3],bb1[1],bb1[4],bb1[2],bb1[5])
    B.as_slices()
# %%
    bb2 = L.nbrhoods['bbox'][1]
    B2 = BBox(bb2[0],bb2[3],bb2[1],bb2[4],bb2[2],bb2[5])
# %%
    bb.intersects(B)
    bb.intersects(B2)
    pointi = extract_points_mm(point)
    pt_idx = points_lps_mm_to_indices(pointi, lm)
# %%
    for pp in pt_idx:
        pp= tuple(pp)
        print(pp)
        print(B.intersects(pp))
        print("\n",B2.intersects(pp))

# %%
    bb.as_slices()
    bb_limits(bb.as_tuple())


# %%
    mups = cc['markups']
    pts = extract_points_mm(cc)
    idx = points_lps_mm_to_indices(pts, img)

# %%
    mrk_path=p
    payload = load_json(mrk_path)
# %%
    pts = extract_points_mm(payload)
    pts = extract_points_mm(cc)
    cs = _detect_coord_system(payload)
    if cs == "RAS":
        pts = _ras_to_lps(pts)
# %%
    bb = bbox_from_markup_file(cc_fn, lm_fn)
# %%

    lms = []
    key = {}
    start_ind = 0
    labels_org = get_labels(L.lm_org)
    if len(labels_org) > 0:
        for label in labels_org:
            lm1, labs = remap_single_label(L.lm_org, label, start_ind)
            k = {l: label for l in labs}
            start_ind = max(labs)
            lms.append(lm1)
            key.update(k)
        merger = sitk.MergeLabelMapFilter()
        merger.SetMethod(0)
        L.lm_cc = merger.Execute(*lms)
        L.key = key
# %%
#SECTION:-------------------- TRoubleshooting--------------------------------------------------------------------------------------
    lm = sitk.ConnectedComponent(L.lm_org)
    get_labels(lm)
# %%
