# %%
from label_analysis.common import *
import logging
import os
import sys
# sys.path.append("/home/ub/code/label_analysis/label_analysis/cpp/build")
# import printcpp
from label_analysis.helpers import remap_single_label
from label_analysis.radiomics import *

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr
from utilz.helpers import *
from utilz.imageviewers import *

from label_analysis.helpers import *

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def make_zero_df(colnames):
    df_ln = len(colnames)
    vals_all = []
    vals_all.append(
        [
            0,
        ]
        * df_ln
    )
    df = pd.DataFrame(vals_all, columns=colnames)
    return df


class LabelMapGeometry(GetAttr):
    """
    lm: can have multiple labels
    img: greyscale sitk image. Default: None. If provided, processes per-label radiomics
    geometry: The centroid is calculated wrt the LPS space (note: slicer uses RAS space or both spaces variably in  the dataprobe to avoid minus). \\
    hence the middle number and sometimes the left number may have reverse sign
    """

    _default = "fil"

    def __init__(self, lm: Union[sitk.Image, str, Path], ignore_labels=[], img=None):

        # printcpp.say("HO")
        if isinstance(lm, Path) or isinstance(lm, str):
            self.lm_fn = lm
            lm = sitk.ReadImage(lm)
        else:
            self.lm_fn = None
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.fil.ComputeFeretDiameterOn()
        if len(ignore_labels) > 0:
            remove_labels = {l: 0 for l in ignore_labels}
            lm = relabel(lm, remove_labels)
        self.img = img
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
        columns = [
            "label",
            "label_cc",
            "cent",
            "bbox",
            "flatness",
            "rad",
            "length",
            "volume",
        ]
        vals_all = []
        if hasattr(self, "key"):
            for key, value in self.key.items():
                centroid = self.GetCentroid(key)
                bbox = self.GetBoundingBox(key)
                flatness = self.GetFlatness(key)
                vals = [
                    value,
                    key,
                    centroid,
                    bbox,
                    flatness,
                    self.ferets[key] / 2,
                    self.ferets[key],
                    self.volumes[key],
                ]
                vals_all.append(vals)
            self.nbrhoods = pd.DataFrame(data=vals_all, columns=columns)
        else:
            self.nbrhoods = make_zero_df(columns)

    def dust(self, dusting_threshold, remove_flat=True):
        if not self.is_empty():
            # dust below length threshold
            inds_small = [l < dusting_threshold for l in self.lengths.values()]
            self.labels_small = list(il.compress(self.labels, inds_small))
            # self.labels_flat =
            self.remove_labels(self.labels_small)
            if remove_flat == True and len(self.nbrhoods) > 0:
                labs_flat = self.nbrhoods["label_cc"][
                    self.nbrhoods["flatness"] == 0
                ].tolist()
                print("Removing flat (2D) labels: ")
                self.remove_labels(labs_flat)

    def remove_labels(self, labels):
        dici = {x: 0 for x in labels}
        print("Removing labels {0}".format(labels))
        self._relabel(dici)
        self.nbrhoods = self.nbrhoods[~self.nbrhoods["label_cc"].isin(labels)]
        self.nbrhoods.reset_index(inplace=True, drop=True)
        for l in labels:
            del self.key[l]
        logging.warning("Neighbourhoods adjusted. {0} removed".format(labels))
        if self.is_empty():
            self.nbrhoods = make_zero_df(self.nbrhoods.columns)

    def execute_filter(self):
        self.lm_cc = to_int(self.lm_cc)
        self.fil.Execute(self.lm_cc)

    def _relabel(self, remapping):
        remapping = np_to_native_dict(remapping)
        self.lm_cc = to_label(self.lm_cc)
        self.lm_cc = sitk.ChangeLabelLabelMap(self.lm_cc, remapping)
        self.execute_filter()
        logging.warning(
            "Labelmap labels have been changed. The nbrhoods df is still as before"
        )

    def radiomics(self, params_fn=None):
        labs_cc = self.nbrhoods["label_cc"]
        if len(labs_cc) > 0:
            rads = radiomics_multiprocess(
                self.img, self.lm_cc, labs_cc, self.lm_fn, params_fn=params_fn
            )
            mini_df = pd.DataFrame(rads)
            self.nbrhoods = self.nbrhoods.merge(
                mini_df, left_on="label_cc", right_on="label"
            )
        else:
            print("No labels found in labelmap. Radiomics not computed")

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
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>

    from fran.managers.datasource import _DS

    DS = _DS()
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-940_fixed_mc")
    test_lms_fldr = Path("/s/xnat_shadow/crc/lms")
    train_fldrs = (
        DS.lits["folder"],
        DS.litq["folder"],
        DS.drli["folder"],
        DS.litqsmall["folder"],
    )
    train_lms_fldrs = [Path(fldr) / ("lms") for fldr in train_fldrs]
    train_img_fldrs = [Path(fldr) / ("images") for fldr in train_fldrs]
    train_img_fns = [list(fldr.glob("*")) for fldr in train_img_fldrs]
    train_img_fns = list(il.chain.from_iterable(train_img_fns))
    train_lm_fns = [list(fldr.glob("*")) for fldr in train_lms_fldrs]
    train_lm_fns = list(il.chain.from_iterable(train_lm_fns))

    test_imgs_fldr = Path("/s/xnat_shadow/crc/images")
    pred_fns = list(preds_fldr.glob("*"))
    test_lm_fns = list(test_lms_fldr.glob("*"))
    test_img_fns = list(test_imgs_fldr.glob("*"))
# %%
    lesions_liver_fn = Path("/home/ub/Documents/litq_10_liver_lesions.nrrd")
    lesions_lung_fn = Path("/home/ub/Documents/litq_10_lung_lesions.nrrd")
    lungs_fn = Path("/home/ub/Documents/litq_10_lungs.nrrd")
    liver_fn = Path("/home/ub/Documents/litq_10_liver.nrrd")
# %%
    Liv = LabelMapGeometry(liver_fn)
    LivLesions = LabelMapGeometry(lesions_liver_fn)

    outfldr = Path("/s/insync/startup/screencasts")
    LivLesions.nbrhoods.to_csv(outfldr / ("liverlesions.csv"))

# %%
    Lungs = LabelMapGeometry(lungs_fn)
    LungLesions = LabelMapGeometry(lesions_lung_fn)
    LungLesions.nbrhoods.to_csv(outfldr / ("lunglesions.csv"))
    Lungs.nbrhoods.to_csv(outfldr / "lungs.csv")
# SECTION:-------------------- RADIOMICS-------------------------------------------------------------------------------------- <CR>

# %%
    fn = "/s/xnat_shadow/crc/lms/crc_CRC211_20170724_AbdoPelvis1p5.nii.gz"
    LG = LabelMapGeometry(fn)
    LG.nbrhoods[:4]
    rows[:4]
# %%
    indices = range(len(train_lm_fns))
    for ind in indices:
        gt_fn = train_lm_fns[ind]
        img_fn = find_matching_fn(gt_fn, train_img_fns)
# %%
# %%
    # fldr=Path("/home/u/code/label_analysis/label_analysis/results/")
    # fns = list(Path("/home/u/code/label_analysis/label_analysis/results/").glob("*csv"))
    # dfs = [pd.read_csv(fn) for fn in fns]
    # df2 = pd.concat(dfs,ignore_index=True)
    # df2.drop_duplicates(inplace=True)
    # len(df2)
    # df2.to_csv(fldr/("final.csv"))
    # fns = df2['fn']
    # fns = [Path(fn) for fn in fns]
    # fns = set(fns)
    #
    # left = set(test_lm_fns).difference(fns)
    # len(left)
# %%
    hoods = []
    for ind in indices:
        gt_fn = train_lm_fns[ind]
        # gt_fn = test_lm_fns[ind]
        img_fn = find_matching_fn(gt_fn, train_img_fns)
        img = sitk.ReadImage(img_fn)
        L = LabelMapGeometry(gt_fn, [1], img)

        if not L.is_empty():
            L.dust(3)
            L.radiomics()
        else:
            L.nbrhoods["fn"] = gt_fn

        hoods.append(L.nbrhoods)
        if ind % 15 == 0:
            dff = pd.concat(hoods, ignore_index=True)
            dff.to_csv("label_analysis/results/hoods{}.csv".format(str(ind)))
# %%

# %%

# %%
# SECTION:-------------------- TROUBLESHOOTJko-------------------------------------------------------------------------------------- <CR>
# %%
    L.nbrhoods
# %%
    for lab in labs_cc:
        print("=" * 20)
        print(lab)
        print(L.fil.GetFlatness(lab))
# %%
    labs_cc = L.nbrhoods["label_cc"]
    params_fn = None
    for lab in labs_cc:
        rads = do_radiomics(L.img, L.lm_cc, lab, L.lm_fn, paramsFile=params_fn)
# %%
    # rads = radiomics_multiprocess(L.img,L.lm_cc,labs_cc,L.lm_fn,params_fn = params_fn)
    # mini_df = pd.DataFrame(rads)
    # L.nbrhoods= L.nbrhoods.merge(mini_df,left_on='label_cc',right_on='label')

# %%
# %%

    # results_df["case444"] = results_df["pred_fn"].apply(
    #     lambda x: info_from_filename(Path(x).name)["case_id"]
    # )

# %%

    gt_fns.sort(key=os.path.getmtime, reverse=True)
    fn = gt_fns[0]
    fn = (
        "/s/xnat_shadow/crc/lms_manual_final/crc_CRC138_20180812_Abdomen3p0I30f3.nii.gz"
    )
    lm = sitk.ReadImage(fn)

    L = LabelMapGeometry(lm)
# %%
