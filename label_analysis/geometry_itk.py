# %%
from dataclasses import dataclass, field
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
from utilz.cprint import cprint
from utilz.fileio import maybe_makedirs
from utilz.helpers import set_autoreload

set_autoreload()

from utilz.imageviewers import ImageMaskViewer
from utilz.itk_sitk import (
    ConvertItkImageToSimpleItkImage,
    ConvertSimpleItkImageToItkImage,
    monai_to_sitk_image,
)

from label_analysis.geometry import (
    LabelMapGeometry,
    load_labelmap_like_to_sitk,
    make_zero_df,
)
from label_analysis.helpers import *
from label_analysis.helpers import to_label

sys.path.append("/home/ub/code/slicer_cpp/protot/build/debug/")
LabelT = itk.UC
Dim = 3
U8 = itk.UC
U8Img = itk.Image[U8, Dim]

pd.set_option("display.expand_frame_repr", False)

@dataclass
class LabelObj:
    ind: int
    obj: itk.StatisticsLabelObject[itk.UL, 3]
    lm: itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]]
    label_org: int
    lm_key: int

    def relabel(self, new_label: int):
        self.obj.SetLabel(new_label)

@dataclass
class LabelObjSet:
    objs: list[LabelObj]
    inds: list[int]
    _map: dict[int, LabelObj] = field(init=False, repr=False)

    def __post_init__(self):
        self._map = {o.ind: o for o in self.objs}

    def __getitem__(self, ind):
        return self._map[ind]

    def remove(self, ind: int):
        idx = self.inds.index(ind)
        self.inds.pop(idx)
        self.objs.pop(idx)
        del self._map[ind]


def relabel_lmap(lmap,remap):
    F = itk.ChangeLabelLabelMapFilter[lmap]
    f = F.New(Input=lmap)
    f.SetChangeMap(remap)   # 9 -> 0 removes label 9
    f.Update()
    lmap2 = f.GetOutput()
    return lmap2

def label_to_labelmap(li_org, lab, compute_feret=True):
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
    BT = itk.BinaryThresholdImageFilter[type(li), U8Img]
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
    # unique labels are a composite of
    # nbrhoods is a first class citizen  - centre of every ops
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
        self._compute_feret = compute_feret
        self.create_li_binary()
        self.create_li_cc()  # creates ordered labelmap from original labels and a key mapping
        self.calc_geom()

    def create_li_binary(self):
        self.li_binary = create_binary_image(self.li_org, 1)

    @property
    def compute_feret(self):
        return self._compute_feret

    def create_li_cc(self, compute_feret=None):
        if compute_feret is not None:
            self._compute_feret = compute_feret
        self.key = {}
        self.unique_lms = []
        self.unique_lms_index_by_label_org = {}
        label_objs = []
        labels_org = get_labels_itk(self.li_org)
        self.li_cc = _alloc_cc_image(self.li_org)
        label_cont = 1
        for lab in labels_org:
            lmap = label_to_labelmap(self.li_org, lab, self.compute_feret)
            remap ={}
            for i in range(lmap.GetNumberOfLabelObjects()):
                obj = lmap.GetNthLabelObject(i)
                lm_key = int(obj.GetLabel())
                remap[lm_key] = label_cont
                self.key[label_cont] = lab
                label_objs.append(
                    LabelObj(
                        ind=label_cont,
                        obj=obj,
                        lm=lmap,
                        label_org=int(lab),
                        lm_key=lm_key,
                    )
                )
                label_cont += 1
            lmap = relabel_lmap(lmap, remap)
            self.unique_lms.append(
                {
                    "lmap": lmap,
                    "label_org": lab,
                    "n_islands": lmap.GetNumberOfLabelObjects(),
                }
            )
            self.unique_lms_index_by_label_org[int(lab)] = len(self.unique_lms) - 1
            self.li_cc = _merge_labelmap_into(self.li_cc, lmap)
        self.label_objs = LabelObjSet(
            objs=label_objs,
            inds=[label_obj.ind for label_obj in label_objs],
        )


    def remap_li_cc(self, remapping):
        F = itk.ChangeLabelImageFilter[self.li_cc,self.li_cc].New(self.li_cc)
        for old, new in remapping.items():
            F.SetChange(old, new)
        F.Update()
        li_cc_new = F.GetOutput()
        return li_cc_new


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
            self.nbrhoods["label_org"] = pd.Series(dtype="int64")
            self.nbrhoods["rad"] = pd.Series(dtype="float64")
            self.nbrhoods["feret"] = pd.Series(dtype="float64")
            self.nbrhoods["volume"] = pd.Series(dtype="float64")
            return

        self.nbrhoods["bbox"] = self.nbrhoods["bbox"].apply(region_to_flat)
        self.nbrhoods["volume"] = self.nbrhoods["volume_cc"]
        self.nbrhoods["label_org"] = self.nbrhoods["label_org"].astype("Int64")


    def _relabel_li_cc(self, remapping, verbose=True):
        li_cc_sitk = ConvertItkImageToSimpleItkImage(self.li_cc, sitk.sitkUInt8)
        li_cc_sitk = relabel(li_cc_sitk, remapping)
        self.li_cc = ConvertSimpleItkImageToItkImage(li_cc_sitk, itk.US)
        if verbose == True:
            logging.warning(
                "Labelmap labels have been changed. The nbrhoods df is still as before"
            )

    def dust(self, dusting_threshold, remove_flat=True):  # in geom
        if not self.is_empty():
            size_metric = "major_axis" if self.compute_feret == False else "feret"
            inds_small = self.nbrhoods[size_metric] < dusting_threshold
            if remove_flat == True:
                inds_flat = self.nbrhoods["flatness"] == 0
                inds_small = inds_small | inds_flat
            inds_small_final = self.nbrhoods.loc[inds_small].index
            self.remove_labelobjects(inds_small_final)
            self.nbrhoods = self.nbrhoods.drop(inds_small_final)

    def remove_labels_from_df(self, labels):
        labels = listify(labels)
        remapping = {x: 0 for x in labels}
        print("Removing labels {0}".format(labels))
        self._relabel_li_cc(remapping)
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

    def remove_labelobjects(self, df_inds):
        if len(df_inds) > 0:
            nbr_tmp = self.nbrhoods.iloc[df_inds].copy()
            for ind, row in nbr_tmp.iterrows():
                label_cc = int(row["label_cc"])
                label_obj = self.label_objs[label_cc]
                label_org = int(label_obj.label_org)
                lm_ind = self._get_unique_lms_index(label_org)
                lmap = self.unique_lms[lm_ind]["lmap"]
                lmap.RemoveLabel(label_obj.lm_key)
                self.label_objs.remove(label_cc)
                dici = {
                    "lmap": lmap,
                    "label_org": label_org,
                    "n_islands": lmap.GetNumberOfLabelObjects(),
                }
                self.unique_lms[lm_ind] = dici
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

    def __len__(self):
        return len(self.nbrhoods)

    @property
    def li_binary_sitk(self):
        return ConvertItkImageToSimpleItkImage(self.li_binary, sitk.sitkUInt8)

    @property
    def li_cc_sitk(self):
        return ConvertItkImageToSimpleItkImage(self.li_cc, sitk.sitkUInt32)

    @property
    def labels(self):
        if self.nbrhoods.empty:
            return []
        return self.nbrhoods.label_cont.tolist()

    @property
    def length(self):
        if self.compute_feret == False:
            lngth = self.nbrhoods["major_axis"].astype(float).to_dict()
            return lngth
        else:
            return self.ferets

    @property
    def ferets(self):
        if self.nbrhoods.empty:
            self._ferets = {}
        elif self.compute_feret == False:
            cprint("compute_feret=False. Using major_axis insetead of feret", "yellow")
            return self.lengths
        else:
            self._ferets = self.nbrhoods["feret"].astype(float).to_dict()
        return self._ferets

    @property
    def lengths(self):
        if self.nbrhoods.empty:
            self._lengths = {}
        else:
            self._lengths = self.nbrhoods["major_axis"].astype(float).to_dict()
        return self._lengths

    @property
    def volumes(self):
        if self.nbrhoods.empty:
            return {}
        return self.nbrhoods["volume_cc"].astype(float).to_dict()

    @property
    def volume_total(self):
        return (
            float(self.nbrhoods["volume_cc"].sum()) if not self.nbrhoods.empty else 0.0
        )


class BBoxInfoFromITK(LabelMapGeometryITK):
    """
    Fast read-only class for nbrhoods. Do not attempt to use other methods from parent classes, not guaranteed to work
    """

    def __init__(
        self,
        li,
        ignore_labels=[],
    ):
        if isinstance(li, str | Path):
            li = itk.imread(li)
        self.li_org = li
        self.ignore_labels = ignore_labels
        self.create_li_cc(False)
        self.calc_geom()
        self.nbrhoods = self.nbrhoods[
            ~self.nbrhoods["label_org"].isin(self.ignore_labels)
        ]


# %%

# SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR>
if __name__ == "__main__":
# %%
    fns = [
        "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_140_Ta70413_ABDOMEN_2p00.nii.gz",
        "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_n1_Ta80605_CAP1p5mm.nii.gz",
    ]

    gt_fn = Path("/Users/ub/datasets/kits23/lms/kits23_00568.nii.gz")
    outfldr = Path("/Users/ub/datasets/tmp")
    maybe_makedirs(outfldr)
    out_fn = outfldr / (gt_fn.name)

    org_fn = outfldr / ("org.nii.gz")
    sitk_fn = outfldr / ("kits23_00568_sitk.nii.gz")
    gt_fn = Path('/media/UB/datasets/kits23/lms/kits23_00121.nii.gz')
    pred_fn = Path('/s/fran_storage/predictions/kits2/KITS2-bah/kits23_00121.nii.gz')
# %%
    L = LabelMapGeometryITK(pred_fn, ignore_labels=[1], compute_feret=False)
    L.nbrhoods
    L.unique_lms
    L.dust(1)
    L.li_cc
    L.li_binary
    get_labels_itk(L.li_binary)
# %%
# %%

    lm = L.unique_lms[1]
    # dusting_threshold = 1
    # L.dust(dusting_threshold)
    l2 = lm['lmap']
    n_label_objects = l2.GetNumberOfLabelObjects()
    print(n_label_objects)
    l2.RemoveLabel(7)


# %%
    for i in range(n_label_objects):
        lo =l2.GetNthLabelObject(i) 
        print(lo.GetLabel())
    # lo.SetLabel(11)
# %%
# %%
    itk.imwrite(L.li_org, org_fn)
    itk.imwrite(L.li_cc, out_fn)
    sitk.WriteImage(L.li_sitk, sitk_fn)
# %%
    L.dust(1)
    L.labels
    get_labels(L.li_cc_sitk)
    get_labels_itk(L.li_cc)
# %%
    cc = L.li_cc
    org = L.li_org
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
    from label_analysis.helpers import get_labels_itk

    get_labels_itk(L.li_cc)

# %%
