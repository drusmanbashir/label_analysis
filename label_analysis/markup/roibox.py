# roi_bbox.py
from __future__ import annotations
from label_analysis.geometry import LabelMapGeometry
from pathlib import Path
from typing import Dict, Any
import json, numpy as np, SimpleITK as sitk
from label_analysis.helpers import to_binary
from label_analysis.markup.core import BBox, load_markups_points_lps, points_lps_mm_to_indices, bbox_from_indices, load_markups_points_indices

class ROIBox:
    """Represents a Slicer ROI of type 'Box' (center/orientation/size in mm).

    Coordinates are taken from the ROI's declared coordinateSystem (LPS or RAS)
    and converted to LPS internally.
    """

    def __init__(self, center_lps: np.ndarray, R_lps: np.ndarray, size_mm: np.ndarray):
        assert center_lps.shape == (3,)
        assert R_lps.shape == (3, 3)
        assert size_mm.shape == (3,)
        self.center = center_lps.astype(float)
        self.R = R_lps.astype(float)  # columns are axis directions in LPS
        self.size = size_mm.astype(float)

    @staticmethod
    def _to_lps_vec(v: np.ndarray, coord_sys: str) -> np.ndarray:
        if coord_sys == "RAS":
            D = np.diag([-1.0, -1.0, 1.0])
            return D @ v
        return v

    @staticmethod
    def _to_lps_mat(M: np.ndarray, coord_sys: str) -> np.ndarray:
        if coord_sys == "RAS":
            D = np.diag([-1.0, -1.0, 1.0])
            return D @ M
        return M

    @classmethod
    def from_markup(cls, markup_obj: Dict[str, Any]) -> "ROIBox":
        if markup_obj.get("type") != "ROI":
            raise ValueError("Markup type must be 'ROI'.")
        if markup_obj.get("roiType", "Box") != "Box":
            raise ValueError("ROI type must be 'Box'.")
        cs = str(markup_obj.get("coordinateSystem", "LPS")).upper()
        if cs not in ("LPS", "RAS"):
            raise ValueError(
                "Unsupported coordinateSystem in ROI: expected 'LPS' or 'RAS'"
            )

        center = np.array(markup_obj["center"], dtype=float)
        size = np.array(markup_obj["size"], dtype=float)  # lengths in mm along ROI axes
        orient = np.array(markup_obj["orientation"], dtype=float).reshape(
            3, 3
        )  # direction cosines

        center_lps = cls._to_lps_vec(center, cs)
        R_lps = cls._to_lps_mat(orient, cs)
        return cls(center_lps, R_lps, size)

    def corners_lps(self) -> np.ndarray:
        """Return the 8 box corner points in LPS mm (shape: 8×3)."""
        hx, hy, hz = 0.5 * self.size
        # Half-axes in ROI-local frame assembled as columns
        H = np.stack(
            [
                np.array([hx, 0.0, 0.0]),
                np.array([0.0, hy, 0.0]),
                np.array([0.0, 0.0, hz]),
            ],
            axis=1,
        )  # (3,3)
        A = self.R @ H  # world (LPS) half-axes as columns
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )  # (8,3)
        return self.center[None, :] + signs @ A.T  # (8,3)

    def to_bbox(self, img: sitk.Image, pad: int = 0) -> BBox:
        """Convert the ROI box to a voxel-space BBox against the given image."""
        pts = self.corners_lps()
        idx = points_lps_mm_to_indices(pts, img)
        return bbox_from_indices(idx, img.GetSize(), pad=pad)

def bbox_from_roi_file(mrk_path: Path, image_path: Path, pad: int = 0) -> BBox:
    payload: Dict[str, Any] = json.loads(Path(mrk_path).read_text())
    roi = ROIBox.from_markup((payload.get("markups") or [])[0])
    img = sitk.ReadImage(str(image_path))
    return roi.to_bbox(img, pad=pad)
# %%
if __name__ == '__main__':

    lm_fn = "/s/fran_storage/predictions/nodes/LITS-1288/nodes_25_20201216_CAP1p5SoftTissue.nii.gz"
    img_fn = "/s/xnat_shadow/nodes/images_pending/thin_slice/images/nodes_25_20201216_CAP1p5SoftTissue.nii.gz"
    roi_fn = "/home/ub/Documents/nodes_25_20201216_CAP1p5SoftTissueR.mrk.json"
    roi2_fn = "/home/ub/Documents/nodes_25_20201216_CAP1p5SoftTissueR_1.mrk.json"
    point_fn = "/home/ub/Documents/nodes_25_20201216_CAP1p5SoftTissue_2_singlepoint.mrk.json"
    L = LabelMapGeometry(lm_fn)
    L.nbrhoods 
# %%
    B1 = bbox_from_roi_file(roi_fn, lm_fn)
    B2 = bbox_from_roi_file(roi2_fn, lm_fn)
    point = load_markups_points_indices(point_fn,lm_fn)
    bboxes = L.nbrhoods['bbox']
    bboxes = [BBox(b[0],b[3],b[1],b[4],b[2],b[5]) for b in bboxes]
    overlaps = [B1.intersects(bbox) for bbox in bboxes]
    overlaps2= [B2.intersects(bbox) for bbox in bboxes]
    overlaps3 =[bbox.intersects(point[0]) for bbox in bboxes]
    overlaps_fina = [any([o1 ,o2,o3]) for o1, o2, o3 in zip(overlaps, overlaps2, overlaps3)]
    rem = L.nbrhoods['label_cc'][overlaps_fina]
    L.remove_labels(rem)
    lm_fn_out = lm_fn.replace(".nii.gz", "_2.nii.gz")
    lm_bin = to_binary(L.lm_cc)
    sitk.WriteImage(lm_bin, lm_fn_out)

    results: Dict[Path, BBox] = {}
    rows: List[Dict[str, Any]] = []
    pts = load_markups_points_lps(cc_fn)
    idx = points_lps_mm_to_indices(pts, lm)
    bb = bbox_from_indices(idx, lm.GetSize(), pad=0)
# %%
    si, sj, sk = bb.as_slices()
    rows.append({
    "file": str(lm_fn),
    "imin": bb.imin, "isize": bb.isize,
    "jmin": bb.jmin, "jsize": bb.jsize,
    "kmin": bb.kmin, "ksize": bb.ksize,
    "slice_i": f"{si.start}:{si.stop}",
    "slice_j": f"{sj.start}:{sj.stop}",
    "slice_k": f"{sk.start}:{sk.stop}",
    })

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
    L.nbrhoods


# %%

if __name__ == "__main__":
    roi_fn = "/home/ub/Documents/nodes_25_20201216_CAP1p5SoftTissueR.mrk.json"
    R = bbox_from_roi_file(roi_fn, image_path="/s/fran_storage/predictions/nodes/LITS-1288/nodes_25_20201216_CAP1p5SoftTissue.nii.gz")
    markup = load_json(roi_fn)
    M = markup["markups"]
    R = ROIBox.from_markup(M[0])

