# %%
import logging
import re
import itertools as il
import sys
from pathlib import Path
from time import time
from typing import Union

import itk
import networkx as nx
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from utilz.helpers import set_autoreload
from utilz.imageviewers import ImageMaskViewer
from utilz.itk_sitk import (ConvertItkImageToSimpleItkImage,
                            ConvertSimpleItkImageToItkImage,
                            monai_to_sitk_image)

from label_analysis.geometry import (LabelMapGeometry,
                                     load_labelmap_like_to_sitk, make_zero_df)
from label_analysis.helpers import *
from label_analysis.helpers import to_label

sys.path.append("/home/ub/code/slicer_cpp/protot/build/debug/")
LabelT = itk.UC
Dim = 3
U8 = itk.UC
U8Img = itk.Image[U8, Dim]


def _label_to_labelmap(li_org, lab, compute_feret=True):
    """
    Extract a single original label from an ITK label image and convert it
    into a ShapeLabelMap, where each connected component becomes one label
    object with geometry (centroid, feret, volume, etc.).
    """
    lm = create_binary_image(li_org, lab, lab)

    LM = itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]]
    Bin2S = itk.BinaryImageToShapeLabelMapFilter[itk.Image[itk.UC, 3], LM]

    f = Bin2S.New(Input=lm)
    f.SetInputForegroundValue(1)
    f.SetComputeFeretDiameter(compute_feret)
    f.Update()

    return f.GetOutput()


def _merge_labelmap_into(out, lmap):
    CCImg = type(out)
    LM = type(lmap)

    to_img = itk.LabelMapToLabelImageFilter[LM, CCImg].New(Input=lmap)
    to_img.Update()

    maxf = itk.MaximumImageFilter[CCImg, CCImg, CCImg].New(
        Input1=out, Input2=to_img.GetOutput()
    )
    maxf.Update()
    return maxf.GetOutput()


def _alloc_cc_image(ref):
    """
    Allocate an empty ITK label image (UL,3) matching the geometry
    of a reference image, initialised to zero.
    """
    CCImg = itk.Image[itk.US, 3]
    out = CCImg.New(
        Regions=ref.GetLargestPossibleRegion(),
        Spacing=ref.GetSpacing(),
        Origin=ref.GetOrigin(),
        Direction=ref.GetDirection(),
    )
    out.Allocate()
    out.FillBuffer(0)
    return out


def create_binary_image(li, low, high=None):
    BT = itk.BinaryThresholdImageFilter[itk.Image[LabelT, Dim], U8Img]
    eq = BT.New(Input=li)
    eq.SetLowerThreshold(low)
    if high:
        eq.SetUpperThreshold(high)
    eq.SetOutsideValue(0)
    eq.SetInsideValue(1)
    eq.Update()
    li_bin = eq.GetOutput()
    return li_bin


def region_to_flat(region):
    return list(region.GetIndex()) + list(region.GetSize())


class LabelMapGeometryITK(LabelMapGeometry):
    def __init__(
        self,
        li: Union[itk.Image, sitk.Image, str, Path],
        ignore_labels=[],
        img=None,
        compute_feret=True,
    ):

        self.unique_lms = []
        if isinstance(li, itk.Image):
            self.li_fn = None
            self.li_org = li
            self.li_sitk = ConvertItkImageToSimpleItkImage(li, sitk.sitkUInt8)
        else:
            li_sitk, src = load_labelmap_like_to_sitk(li)
            self.li_fn = src
            self.li_sitk = li_sitk
            self.li_org = ConvertSimpleItkImageToItkImage(li_sitk, itk.UC)
        # self.lm.Update()
        assert isinstance(ignore_labels, list | None), "ignore_labels must be a list"
        if len(ignore_labels) > 0:
            print("Removing labels {0}".format(ignore_labels))
            remove_labels = {l: 0 for l in ignore_labels}
            self.li_org = relabel_itk(self.li_org, remove_labels)
            self.li_sitk = relabel(self.li_sitk, remove_labels)
        self.img = img

        self.create_li_binary()
        self.create_li_cc(
            compute_feret=compute_feret
        )  # creates ordered labelmap from original labels and a key mapping
        self.calc_geom()

    def create_li_binary(self):
        self.li_binary = create_binary_image(self.li_org, 1)


    def create_li_cc(self, compute_feret=True):
        self.key = {}
        self.unique_lms = []
        self.unique_lms_index_by_label_org = {}

        labels_org = get_labels(self.li_sitk)
        self.li_cc = _alloc_cc_image(self.li_org)

        if not labels_org:
            return

        for lab in labels_org:
            lmap = _label_to_labelmap(self.li_org, lab, compute_feret)

            for cc in lmap.GetLabels():
                self.key[int(cc)] = lab

            self.unique_lms.append(
                {
                    "lmap": lmap,
                    "label_org": lab,
                    "n_islands": lmap.GetNumberOfLabelObjects(),
                }
            )
            self.unique_lms_index_by_label_org[int(lab)] = len(self.unique_lms) - 1

            self.li_cc = _merge_labelmap_into(self.li_cc, lmap)

    def calc_geom(self):
        columns = [
            "label_org",
            "label_cc",
            "cent",
            "bbox",
            "flatness",
            "rad",
            "feret",
            "major_axis",
            "minor_axis",
            "least_axis",
            "volume_cc",
        ]
        rows = []
        for lm_dict in self.unique_lms:
            label_org = lm_dict["label_org"]
            n_islands = lm_dict["n_islands"]
            lm = lm_dict["lmap"]
            for i in tqdm(range(n_islands)):
                obj = lm.GetNthLabelObject(i)
                mom = obj.GetPrincipalMoments()  # tuple of eigenvalues
                mom = np.asarray(obj.GetPrincipalMoments(), dtype=np.float64)

                # If they're substantially negative, that's not just rounding noise
                if np.any(mom < -1e-6):
                    logging.warning(
                        "Negative principal moments (label_org=%s, label_cc=%s): %s",
                        label_org,
                        int(obj.GetLabel()),
                        mom,
                    )

                mom = np.maximum(mom, 0.0)  # clamp tiny negatives to 0
                major_m, mid_m, least_m = (
                    float(mom.max()),
                    float(np.median(mom)),
                    float(mom.min()),
                )

                major_axis = 4.0 * np.sqrt(major_m)
                minor_axis = 4.0 * np.sqrt(mid_m)
                least_axis = 4.0 * np.sqrt(least_m)
                rows.append(
                    {
                        "label_org": label_org,
                        "label_cc": int(obj.GetLabel()),
                        "cent": tuple(obj.GetCentroid()),
                        "bbox": obj.GetBoundingBox(),
                        "flatness": float(obj.GetFlatness()),
                        "rad": float(obj.GetEquivalentSphericalRadius()),
                        "feret": float(obj.GetFeretDiameter()),
                        "major_axis": float(major_axis),
                        "minor_axis": float(minor_axis),
                        "least_axis": float(least_axis),
                        "volume_cc": float(obj.GetPhysicalSize()) * 1e-3,
                    }
                )

        self.nbrhoods = pd.DataFrame(rows, columns=columns)
        if self.nbrhoods.empty:
            self.nbrhoods["label"] = pd.Series(dtype="int64")
            self.nbrhoods["rad"] = pd.Series(dtype="float64")
            self.nbrhoods["feret"] = pd.Series(dtype="float64")
            self.nbrhoods["volume"] = pd.Series(dtype="float64")
            return

        self.nbrhoods["bbox"] = self.nbrhoods["bbox"].apply(region_to_flat)
        self.nbrhoods["label"] = self.nbrhoods["label_org"].astype(int)
        self.nbrhoods["volume"] = self.nbrhoods["volume_cc"]

    def _relabel(self, remapping, verbose=True):
        li_cc_sitk = ConvertItkImageToSimpleItkImage(self.li_cc, sitk.sitkUInt8)
        li_cc_sitk= relabel(li_cc_sitk, remapping)
        self.li_cc = ConvertSimpleItkImageToItkImage(li_cc_sitk, itk.US)
        if verbose==True:
            logging.warning(
                "Labelmap labels have been changed. The nbrhoods df is still as before"
            )



    def dust(self, dusting_threshold, remove_flat=True): # in geom
        if not self.is_empty():
            # dust below length threshold
            inds_small = [l < dusting_threshold for l in self.ferets.values()]
            self.labels_small = list(il.compress(self.labels, inds_small))
            if remove_flat == True and len(self.nbrhoods) > 0:
                labs_flat = self.nbrhoods["label_cc"][
                    self.nbrhoods["flatness"] == 0
                ].tolist()
                print("Removing flat (2D) labels: ")
                self.remove_labels(labs_flat)


    def remove_labels(self, labels): 
        labels = listify(labels)
        remapping = {x: 0 for x in labels}
        print("Removing labels {0}".format(labels))
        self._relabel(remapping)
        self.nbrhoods = self.nbrhoods[~self.nbrhoods["label_cc"].isin(labels)]
        self.nbrhoods.reset_index(inplace=True, drop=True)
        for l in labels:
            del self.key[l]
        logging.warning("Neighbourhoods adjusted. {0} removed".format(labels))
        if self.is_empty():
            self.nbrhoods = make_zero_df(self.nbrhoods.columns)

    def _get_unique_lms_index(self, label_org):
        try:
            return self.unique_lms_index_by_label_org[int(label_org)]
        except (AttributeError, KeyError) as exc:
            available = [int(lm["label_org"]) for lm in getattr(self, "unique_lms", [])]
            raise IndexError(
                "Could not map label_org={} to self.unique_lms. Available label_org "
                "values: {}. This usually means labels are sparse after filtering or "
                "ignore-label removal, so label_org cannot be used as a positional index.".format(
                    int(label_org), available
                )
            ) from exc

    def remove_rows(self, rows_for_removal):
        if len(rows_for_removal) > 0:
            nbr_tmp = self.nbrhoods[rows_for_removal].copy()
            for ind, row in nbr_tmp.iterrows():
                label_org = row["label_org"]
                lobj_ind = self._get_unique_lms_index(label_org)
                label_cc = int(row["label_cc"])
                lmap = self.unique_lms[lobj_ind]["lmap"]
                lmap.RemoveLabel(label_cc)
                dici = {
                    "lmap": lmap,
                    "label_org": label_org,
                    "n_islands": lmap.GetNumberOfLabelObjects(),
                }
                self.unique_lms[lobj_ind] = dici
            self._create_out_image()

    def _create_out_image(self):
        lis = []
        self.li_cc = _alloc_cc_image(self.li_org)
        for lm1 in self.unique_lms:
            lmap1 = lm1["lmap"]
            self.li_cc = _merge_labelmap_into(self.li_cc, lmap1)
            instance = itk.LabelMapToBinaryImageFilter[
                itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]], itk.Image[itk.UC, 3]
            ].New(lmap1)
            instance.SetForegroundValue(lm1["label_org"])
            instance.Update()
            out = instance.GetOutput()
            lis.append(out)
        if len(lis) > 1:
            self.li_out = itk.AddImageFilter(*lis)
        elif len(lis) == 1:
            self.li_out = lis[0]
        else:
            self.li_out = None


    @property
    def li_binary_sitk(self):
            return ConvertItkImageToSimpleItkImage(self.li_binary, sitk.sitkUInt8)

    @property
    def li_cc_sitk(self):
        return ConvertItkImageToSimpleItkImage(self.li_cc, sitk.sitkUInt32)

    @property
    def labels(self):
        if not hasattr(self, "nbrhoods"):
            labels = []
            for lm_dict in self.unique_lms:
                labels.extend(int(label) for label in lm_dict["lmap"].GetLabels())
            return labels
        zeros = (self.nbrhoods==0).all().all()
        if self.nbrhoods.empty or zeros==True:
            return []
            
        return self.nbrhoods["label_cc"].astype(int).tolist()

    @property
    def length(self):
        return self.ferets

    @property
    def ferets(self):
        if self.nbrhoods.empty:
            self._ferets = {}
        else:
            self._ferets = (
                self.nbrhoods.set_index("label_cc")["feret"].astype(float).to_dict()
            )
        return self._ferets

    @property
    def volumes(self):
        if self.nbrhoods.empty:
            return {}
        return self.nbrhoods.set_index("label_cc")["volume_cc"].astype(float).to_dict()

    @property
    def volume_total(self):
        return (
            float(self.nbrhoods["volume_cc"].sum()) if not self.nbrhoods.empty else 0.0
        )


# %%
# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR>
if __name__ == "__main__":
# %%
    fns = [
        "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_140_Ta70413_ABDOMEN_2p00.nii.gz",
        "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_n1_Ta80605_CAP1p5mm.nii.gz",
    ]


    gt_fn = Path("/s/fran_storage/predictions/kits/KITS-n7/kits23_00209.pt")

# %%
    L = LabelMapGeometryITK(gt_fn, ignore_labels=[])
# %%
    L.dust(1)
    L.labels
    get_labels(L.li_cc_sitk)
    get_labels_itk(L.li_cc)
# %%

    nodes_fldr = Path("/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417")

# %%

    # li0 = "/media/UB/datasets/kits23/lms/kits23_00018.nii.gz"
    li0 = "/media/UB/datasets/lidc/lms/lidc_0143.nii.gz"
    L = LabelMapGeometryITK(li0, ignore_labels=[], dusting_threshold=0.0)
    L.nbrhoods.to_csv("latest.csv")

    itk.imwrite(L.li_cc, "nodes_140_cc_labels.nii.gz")
# %%
    L2 = LabelMapGeometryITK(li2)
    L2.nbrhoods.to_csv("nodes_140_2labels.csv")
# %%
    fldr = Path(
        "/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex050/lms"
    )
    fns = list(fldr.glob("*"))
# %%
    for fn in fns:
        print(fn)
        L = LabelMapGeometryITK(fn)
# %%
    import torch

    li_fn = Path(
        "/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex050/lms/kits23_00290.pt"
    )
    im_fn = Path(
        "/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex050/images/kits23_00290.pt"
    )

    lm = torch.load(li_fn, weights_only=False)
    im = torch.load(im_fn, weights_only=False)

    ImageMaskViewer([im, lm])
    L = LabelMapGeometryITK(lm)
    L.nbrhoods
# %%
    im_sitk, src = monai_to_sitk_image(lm)

    lm = sitk.GetArrayFromImage(im_sitk)

    lm = torch.Tensor(lm)

    itk.imwrite(L.li_cc, "test.nii.gz")
    ImageMaskViewer([im, lm])
    L = LabelMapGeometryITK(lm)
    L.nbrhoods


# %%
