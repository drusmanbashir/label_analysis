# %%
import itertools as il
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.all import store_true
from fastcore.basics import store_attr
from label_analysis.helpers import (astype, get_labels, remove_organ_label,
                                   single_label, to_cc, to_int)
from label_analysis.utils import align_sitk_imgs
from pyxnat.core.resources import shutil

from utilz.fileio import load_json, maybe_makedirs
from utilz.helpers import find_matching_fn, pp
from utilz.imageviewers import view_sitk
from utilz.string import info_from_filename, int_to_str, strip_extension

tr = ipdb.set_trace


def identical_key_vals(dicti):
    return list(dicti.keys()) == list(dicti.values())


class RemapFromDF:
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_fldr: Path,
        target_label: int,
        organ_label: int = None,
        schema=None,
    ):
        '''

        df: The main columns are : lesion_labels which can be 'benign', 'mets' or 'json' 
        dataset_fldr: The root folder. Subfolders required: images , lms, markups
        A single 'target_label' will be remappe to a different label based on the provided schema.


        '''
            
        store_attr()
        self.df = self.df.reset_index()  # prevents errors while concat
        if self.schema is None:
            self.schema = {"normal":2, "benign": 2, "mets": 3} # normal : 2 in case some cases labelled normal have some benign lesions
        self.out_fldr = dataset_fldr / "lms_mc"
        self.lms_fldr = dataset_fldr / "lms"
        self.img_fldr = dataset_fldr / "images"
        assert all([fldr.exists() for fldr in [self.img_fldr, self.lms_fldr]]), "Missing folders (one or both) {0} , {1}".format(self.img_fldr, self.lms_fldr)
        self.img_fns = list(self.img_fldr.glob("*"))
        self.lm_fns = list(self.lms_fldr.glob("*"))
        self.markup_fns = list((dataset_fldr / "markups").glob("*"))

        additional_columns = []
        for row in self.df.itertuples():
            additional_columns.append(self.get_matching_fns(row))

        dft = pd.DataFrame(additional_columns)
        self.df = pd.concat([self.df, dft], axis=1)

        maybe_makedirs(self.out_fldr)

    def process(self):
        for ind in range(len(self.df)):
            row = self.df.iloc[ind]
            if row.lesion_labels == "na":
                pass
            else:
                self.process_row(row)

    def process_row(self, row, overwrite=False):
        case_id = row.case_id
        print("Remapping {} to {}".format(row.lm_fn, row.lm_fn_out))
        if not overwrite and Path(row.lm_fn_out).exists():
            print("File {} exists. Skipping..".format(row.lm_fn_out))
            return
        if row.lesion_labels == "json":
            R = RemapFromMarkup(organ_label=self.organ_label)
            R.process(row.lm_fn, row.lm_fn_out, row.markup_fn)
        elif row.lesion_labels == "normal":
            print("No lesions as per df. Will copy as such to dest folder")
            shutil.copy(row.lm_fn, row.lm_fn_out)
        else:
            case_mapping = {self.target_label: self.schema[row.lesion_labels]}
            if identical_key_vals(case_mapping):
                print(
                    "Identical mapping for {0}' by default. No remapping needed. Making copy in {1}".format(
                        case_id, self.out_fldr
                    )
                )
                shutil.copy(row.lm_fn, row.lm_fn_out)
            else:
                print("Remapping schema: {}".format(case_mapping))
                mask = sitk.ReadImage(str(row.lm_fn))
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                mask = sitk.ChangeLabel(mask, case_mapping)
                sitk.WriteImage(mask, str(row.lm_fn_out))
                print("Done")

    def get_matching_fns(self, row):
        assert row.lesion_labels in [
            "benign",
            "mets",
            "normal",
            "na",
            "json",
        ], "Illegal lesion label: {}".format(row.lesion_labels)
        if self.excluded(row.lesion_labels) == True:
            return {
                "img_fn": None,
                "lm_fn": None,
                "markup_fn": None,
                "lm_fn_out": None,
            }
        else:
            case_id = row.case_id
            # if case_id == 'CRC006':
            #     tr()
            img_fn = [
                fn
                for fn in self.img_fns
                if info_from_filename(fn.name)["case_id"] == case_id
            ]
            assert len(img_fn) == 1, "multiple (or none) images found for {}".format(
                case_id
            )
            img_fn = img_fn[0]
            lm_fn = find_matching_fn(img_fn, self.lm_fns)
            if row.lesion_labels == "json":
                markup_fn = [
                    fn
                    for fn in self.markup_fns
                    if info_from_filename(fn.name)["case_id"] == case_id
                ]
                assert (
                    len(markup_fn) == 1
                ), "multiple (or None) markups found for {}".format(case_id)
                markup_fn = markup_fn[0]
            else:
                markup_fn = None

            lm_fn_out = self.out_fldr / (lm_fn.name)
            dici = {
                "img_fn": img_fn,
                "lm_fn": lm_fn,
                "markup_fn": markup_fn,
                "lm_fn_out": lm_fn_out,
            }
            return dici

    def excluded(self, label):
        label = label.lower()
        if label == "na" or "exclude" in label:
            return True
        else:
            return False


class RemapFromMarkup:
    def __init__(self, organ_label: int = None):
        # organ_label, if specified. This is removed.
        store_attr()
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.fil.SetComputeFeretDiameter(True)

    def preprocess_lm(self, lm):
        self.lm_bkp = lm
        lm_cc = sitk.Image(self.lm_bkp)
        lm_cc = to_int(lm_cc)
        if self.organ_label is not None:
            remove_mapping = {self.organ_label: 0}
            lm_cc = sitk.ChangeLabel(lm_cc, remove_mapping)
        lm_cc = to_cc(lm_cc)
        return lm_cc

    def get_fid_locs(self, slicer_markups):
        mu = slicer_markups["markups"]
        fid_info = mu[0]["controlPoints"]
        fid_locs = [a["position"] for a in fid_info]
        fid_locs = np.array(fid_locs)
        return fid_locs

    def process(self, lm_fn, lm_fn_out, markup_fn, overwrite=False):
        if lm_fn_out.exists() and not overwrite:
            print("File {} exists. Skipping..".format(lm_fn_out))
            return
        markups = load_json(markup_fn)
        markup_type = "mets" if "mets" in markup_fn.name else "benign"
        if markup_type == "mets":
            fid_label, non_fid_label = 3, 2
        else:
            fid_label, non_fid_label = 2, 3

        lm = sitk.ReadImage(lm_fn)
        lm_cc = self.relabel(lm, markups, fid_label, non_fid_label)
        print("Writing {}".format(lm_fn_out))
        sitk.WriteImage(lm_cc, lm_fn_out)

    def relabel(self, lm, fid_markups, fid_label, non_fid_label):
        lm_cc = self.preprocess_lm(lm)
        self.fil.Execute(lm_cc)
        labels = self.fil.GetLabels()
        centroids = [self.fil.GetCentroid(lab) for lab in labels]
        centroids = np.array(centroids)

        radii = [self.fil.GetFeretDiameter(lab) / 2 for lab in labels]

        fid_locs = self.get_fid_locs(fid_markups)
        distance_vecs = np.array([a - centroids for a in fid_locs])
        distances = np.linalg.norm(distance_vecs, axis=2)
        closest_markers = distances < radii
        closest_marker_indices = np.nonzero(closest_markers)[1] + 1
        remapping = {x: non_fid_label for x in labels}
        for key in closest_marker_indices:
            remapping[key] = fid_label

        # assert((len(closest_markers)) == (len(lu:=np.unique(closest_markers)))), "Repeat values Some fiducials are closest to more than one lesion. It may be an orphan or from other category"
        lm_cc = sitk.ChangeLabel(lm_cc, remapping)
        if self.organ_label is not None:
            lm_cc = self.put_organ_back(lm_cc)
        return lm_cc

    def put_organ_back(self, lm_cc):
        organ = sitk.GetArrayFromImage(self.lm_bkp)
        lesions_cc = sitk.GetArrayFromImage(lm_cc)

        lesions_cc[organ == 1] = 1
        lm_cc = sitk.GetImageFromArray(lesions_cc)
        lm_cc = align_sitk_imgs(lm_cc, self.lm_bkp)
        del self.lm_bkp
        return lm_cc


# %%

# %% [markdown]
## AIscreening dataset
if __name__ == "__main__":
# %%
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images/")

    # df_fn = Path("/s/datasets_bkp/lits_segs_improved/segs_notes.csv")
    df_fn = Path("/s/xnat_shadow/crc/images_more/images_more_summary_fake.xlsx")
    df = pd.read_excel(df_fn, sheet_name="Sheet3")
    df.dropna(subset=["case_id", "lesion_labels"], inplace=True)
    fldr = Path("/s/xnat_shadow/crc//")
    # int_to_str(ĳkĳk)

    completed = ["CRC" + int_to_str(x, 3) for x in range(500)]

    df_comp = df.loc[df["case_id"].isin(completed)]

    df_comp = df_comp.reset_index()
# %%
    RD = RemapFromDF(df_comp, fldr, organ_label=None, target_label=3)

    row = df_comp.iloc[14]
    RD.process()
# %%
    info_from_filename(
        Path("/s/datasets_bkp/litqsmall/images/litqsmall_00036.nii.gz").name
    )
# %%
    cid = "CRC018"
    mini = RD.df.loc[RD.df.case_id == cid]
    mini = mini.iloc[1]
    RD.process_row(mini)
# %%
    RD.df = df_comp
    dataset_fldr = fldr
    target_label = 2
    schema = None
    fixed_labels = [0, 1]
# %%
    RD.mapping = {label: label for label in fixed_labels}
    if RD.schema is None:
        RD.schema = {"benign": 2, "mets": 3}
    RD.out_fldr = dataset_fldr / "masks_multiclass"
    RD.lms_fldr = dataset_fldr / "masks"
    RD.img_fldr = dataset_fldr / "images"
    RD.img_fns = list(RD.img_fldr.glob("*"))
    RD.lm_fns = list(RD.lms_fldr.glob("*"))
    RD.markup_fns = list((dataset_fldr / "markups").glob("*"))

    additional_columns = []
    for row in RD.df.itertuples():
        additional_columns.append(RD.get_matching_fns(row))

        RD.df = pd.concat([RD.df, pd.DataFrame(additional_columns)], axis=1)


# %%
