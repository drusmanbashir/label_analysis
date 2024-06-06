import os
# %%
import logging
import sys
import time
from functools import reduce

from label_analysis.helpers import remap_single_label
from label_analysis.utils import is_sitk_file

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr
from label_analysis.helpers import *

from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.string import (find_file, info_from_filename, match_filenames,
                               strip_extension, strip_slicer_strings)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
class LabelMapGeometry(GetAttr):
    """
    lm: can have multiple labels
    img: greyscale sitk image. Default: None. If provided, processes per-label radiomics
    """

    _default = "fil"

    def __init__(self, lm: sitk.Image, ignore_labels=[], img=None):
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.fil.ComputeFeretDiameterOn()
        if len(ignore_labels) > 0:
            remove_labels = {l: 0 for l in ignore_labels}
            lm = relabel(lm, remove_labels)
        self.lm_org = lm
        self.create_lm_binary()
        self.create_lm_cc()  # creates ordered labelmap from original labels and a key mapping
        self.execute_filter()
        self.calc_geom()


    # def calc_bb(self):
    #     self.fil.ComputteBounndin


    def create_lm_binary(self):
        lm_tmp = self.lm_org
        lm_tmp = to_label(lm_tmp)
        self.lm_binary = sitk.LabelMapToBinary(lm_tmp)

    def create_lm_cc(self):
        lms = []
        key = {}
        start_ind = 0
        labels_org = get_labels(self.lm_org)
        if len(labels_org) > 0:
            for label in labels_org:
                lm1, labs = remap_single_label(self.lm_org, label, start_ind)
                k = {l: label for l in labs}
                start_ind = max(labs)
                lms.append(lm1)
                key.update(k)
            merger = sitk.MergeLabelMapFilter()
            merger.SetMethod(0)
            self.lm_cc = merger.Execute(*lms)
            self.key = key
        else:
            print("Empty labelmap")
            self.lm_cc = sitk.Image()

    def calc_geom(self):
        columns = ["label", "label_cc", "cent","bbox", "rad", "length", "volume"]
        vals_all = []
        if hasattr(self, "key"):
            for key, value in self.key.items():
                centroid = self.GetCentroid(key)
                bbox = self.GetBoundingBox(key)
                vals = [
                    value,
                    key,
                    centroid,
                    bbox,
                    self.ferets[key]/2,
                    self.ferets[key],
                    self.volumes[key],
                ]
                vals_all.append(vals)
        else:
            vals_all.append(
                [
                    0,
                ]
                * 7
            )
        self.nbrhoods = pd.DataFrame(data=vals_all, columns=columns)

    def dust(self, dusting_threshold):
        inds_small = [l < dusting_threshold for l in self.lengths.values()]
        self.labels_small = list(il.compress(self.labels, inds_small))
        self._remove_labels(self.labels_small)

    def _remove_labels(self, labels):
        dici = {x: 0 for x in labels}
        print("Removing labels {0}".format(labels))
        self.relabel(dici)
        self.nbrhoods = self.nbrhoods[~self.nbrhoods["label_cc"].isin(labels)]
        self.nbrhoods.reset_index(inplace=True, drop=True)
        for l in labels:
            del self.key[l]

    def execute_filter(self):
        self.lm_cc = to_int(self.lm_cc)
        self.fil.Execute(self.lm_cc)

    def relabel(self, remapping):
        remapping = np_to_native_dict(remapping)
        self.lm_cc = to_label(self.lm_cc)
        self.lm_cc = sitk.ChangeLabelLabelMap(self.lm_cc, remapping)
        self.execute_filter()
        logging.warning("Labelmap labels have been changed. The nbrhoods df is still as before")

    def is_empty(self):
        return True if self.__len__() == 0 else False

    def __str__(self) -> str:
        pass

    @property
    def labels(self):
        return self.GetLabels()

    @property
    def labels_unique(self):
        return list(set(self.key.values()))

    def __len__(self):
        return len(self.labels)

    @property
    def volumes(self):
        self._volumes = {x: self.GetPhysicalSize(x) * 1e-3 for x in self.labels}
        return self._volumes

    @property
    def volume_total(self):
        return sum(self.volumes.values())

    @property
    def lengths(self):
        return self.ferets

    @property
    def ferets(self):
        self._ferets = {x: self.GetFeretDiameter(x) for x in self.labels}
        return self._ferets

    @property
    def centroids(self):
        return [np.array(self.fil.GetCentroid(x)) for x in self.labels]




# %%
if __name__ == "__main__":
    preds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-940_fixed_mc"
    )

# %%
    gt_fldr = Path("/s/xnat_shadow/crc/lms_manual_final")
    gt_fns = list(gt_fldr.glob("*"))
    gt_fns = [fn for fn in gt_fns if is_sitk_file(fn)]

    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    results_df = pd.read_excel(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh0mm_results.xlsx",
        index_col=None,
    )


    # results_df["case444"] = results_df["pred_fn"].apply(
    #     lambda x: info_from_filename(Path(x).name)["case_id"]
    # )

# %%

    gt_fns.sort(key=os.path.getmtime, reverse=True)
    fn = gt_fns[0]
    fn = "/s/xnat_shadow/crc/lms_manual_final/crc_CRC138_20180812_Abdomen3p0I30f3.nii.gz"
    lm = sitk.ReadImage(fn)

    L = LabelMapGeometry(lm)
# %%
