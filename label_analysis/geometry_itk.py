# %%
import re
import logging
import numpy as np
import sys
from pathlib import Path
from time import time
from typing import Union

import itk
import networkx as nx
import numpy as np
import pandas as pd
import SimpleITK as sitk
from utilz.itk_sitk import (ConvertItkImageToSimpleItkImage,
                                 ConvertSimpleItkImageToItkImage)
from tqdm import tqdm

from label_analysis.geometry import LabelMapGeometry
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
        dusting_threshold=0,
        remove_flat=False,
        img=None,
        compute_feret=True,
    ):
        self.unique_lms = []
        # printcpp.say("HO")
        if isinstance(li, Path) or isinstance(li, str):
            self.lm_fn = li
            li = itk.imread(li, itk.UC)
        else:
            self.lm_fn = None

        if isinstance(li, sitk.Image):
            self.li_sitk = li
            self.li_org = ConvertSimpleItkImageToItkImage(li, itk.UC)
        elif isinstance(li, itk.Image):
            self.li_org = li
            self.li_sitk = ConvertItkImageToSimpleItkImage(li, sitk.sitkUInt8)
        # self.lm.Update()
        if len(ignore_labels) > 0:
            self._remove_labels(ignore_labels)
        self.img = img

        self.create_li_binary()
        self.create_lm_cc(
            compute_feret=compute_feret
        )  # creates ordered labelmap from original labels and a key mapping
        self.dust_and_calc_geom(
            dusting_threshold=dusting_threshold, remove_flat=remove_flat
        )

    def remove_labels(self, labels):
        raise NotImplementedError

    def remove_rows(self, rows_for_removal):
        if len(rows_for_removal) > 0:
            nbr_tmp = self.nbrhoods[rows_for_removal].copy()
            for ind, row in nbr_tmp.iterrows():
                label_org = row["label_org"]
                lobj_ind = label_org - 1
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
        for lm1 in self.unique_lms:
            lmap1 = lm1["lmap"]
            instance = itk.LabelMapToBinaryImageFilter[
                itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]], itk.Image[itk.UC, 3]
            ].New(lmap1)
            instance.SetForegroundValue(lm1["label_org"])
            instance.Update()
            out = instance.GetOutput()
            lis.append(out)
        if len(lis) > 1:
            self.li_out = itk.AddImageFilter(*lis)
        else:
            self.li_out = lis[0]

    def dust_and_calc_geom(self, dusting_threshold=0, remove_flat=False):
        self.dust(dusting_threshold=dusting_threshold, remove_flat=remove_flat)
        self.calc_geom()

    def dust(self, dusting_threshold=0, remove_flat=False):
        if not self.is_empty() and any([dusting_threshold > 0, remove_flat]):
            if remove_flat == True:
                rows_for_removal = (self.nbrhoods["flatness"] == 0) | (
                    self.nbrhoods["feret"] < dusting_threshold
                )
            else:
                rows_for_removal = self.nbrhoods["feret"] < dusting_threshold
            self.remove_rows(rows_for_removal)

    def create_li_binary(self):
        self.li_binary = create_binary_image(self.li_org, 1)


    def __len__(self):
        return len(self.unique_lms)

    def calc_geom(self):
        rows = []
        for lm_dict in self.unique_lms:
            label_org = lm_dict["label_org"]
            n_islands = lm_dict["n_islands"]
            lm = lm_dict["lmap"]
            for i in tqdm(range(n_islands)):
                obj = lm.GetNthLabelObject(i)
                mom = obj.GetPrincipalMoments()      # tuple of eigenvalues
                mom = np.asarray(obj.GetPrincipalMoments(), dtype=np.float64)

                # If they're substantially negative, that's not just rounding noise
                if np.any(mom < -1e-6):
                    logging.warning(
                        "Negative principal moments (label_org=%s, label_cc=%s): %s",
                        label_org, int(obj.GetLabel()), mom
                    )

                mom = np.maximum(mom, 0.0)  # clamp tiny negatives to 0
                major_m, mid_m, least_m = float(mom.max()), float(np.median(mom)), float(mom.min())

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
                        "feret": float(obj.GetFeretDiameter()),
                        "major_axis": float(major_axis),
                        "minor_axis": float(minor_axis),
                        "least_axis": float(least_axis),
                        "volume_mm3": float(obj.GetPhysicalSize()),
                    }
                )


        self.nbrhoods = pd.DataFrame(rows)
        self.nbrhoods["bbox"] = self.nbrhoods["bbox"].apply(region_to_flat)



    def create_lm_cc(self, compute_feret=True):
        """
        Build a global connected-component label image (lm_cc) from an input
        multi-label segmentation.

        Each connected component receives a unique label_cc, while `self.key`
        records the mapping: label_cc → original label.
        """
        self.key = {}
        self.unique_lms = []

        labels_org = get_labels(self.li_sitk)
        if not labels_org:
            self.lm_cc = itk.Image[itk.UL, 3]()

        self.lm_cc = _alloc_cc_image(self.li_org)

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

            self.lm_cc = _merge_labelmap_into(self.lm_cc, lmap)

if __name__ == '__main__':
# %%
    nodes_fldr = Path("/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417")
    fns = ["/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_140_Ta70413_ABDOMEN_2p00.nii.gz",
           "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_n1_Ta80605_CAP1p5mm.nii.gz"]

    fn2 = "nodes_140_2labels.nii.gz-label.nrrd"
    li2 = sitk.ReadImage(fn2)

    li0 = sitk.ReadImage(fns[0])
    li1 = sitk.ReadImage(fns[1])
# %%

    L = LabelMapGeometryITK(li0)
    L.nbrhoods.to_csv("latest.csv")

    itk.imwrite(L.lm_cc, "nodes_140_cc_labels.nii.gz")
# %%
    L2 = LabelMapGeometryITK(li2)
    L2.nbrhoods.to_csv("nodes_140_2labels.csv")
