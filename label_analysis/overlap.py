# %%
import os
import ast
from fran.utils.dictopts import key_from_value
import sys
import time
from functools import reduce
from radiomics import featureextractor

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import networkx as nx
import ray
from dicom_utils.capestart_related import find_files_from_list
from label_analysis.geometry import LabelMapGeometry

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr
from label_analysis.helpers import *

from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.string import (
    find_file,
    info_from_filename,
    match_filenames,
    strip_extension,
    strip_slicer_strings,
)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

# %% [markdown]
## this


def fk_generator(start=0):
    key = start
    while True:
        yield key
        key += 1


def keep_largest(onedarray):
    largest = onedarray.max()
    onedarray[onedarray < largest] = 0
    return onedarray


@astype([5, 5], [0, 1])
def labels_overlap(gt_cc, pred_cc, lab_gt, lab_pred, compute_jaccard=True):
    gt_all_labels = get_labels(gt_cc)
    assert (
        lab_gt in gt_all_labels
    ), "Label {} is not present in the Groundtruth ".format(lab_gt)
    mask2 = single_label(gt_cc, lab_gt)
    pred2 = single_label(pred_cc, lab_pred)
    fil = sitk.LabelOverlapMeasuresImageFilter()
    a, b = map(to_int, [mask2, pred2])
    fil.Execute(a, b)
    if compute_jaccard == True:
        dsc, jac = fil.GetDiceCoefficient(), fil.GetJaccardCoefficient()
        return dsc, jac
    else:
        dsc = fil.GetDiceCoefficient()
        return dsc


def append_empty_rows(dataframe, n):
    for i in range(n):
        dataframe.loc[len(dataframe)] = pd.Series(dtype="float64")
    return dataframe


def proximity_indices(df_lm1, df_lm2):
    # returns inds starting at 0
    # each df must have 'rad' and 'cent' rows
    proximity_matrix = np.zeros([len(df_lm1), len(df_lm2)])
    for i in range(len(df_lm1)):
        gl = df_lm1.iloc[i]
        gl_rad, gl_cent = gl.rad, gl.cent
        for j in range(len(df_lm2)):
            pl = df_lm2.iloc[j]
            pl_rad, pl_cent = pl.rad, pl.cent
            pl_cent = np.array(pl_cent)
            dist_vec = pl_cent - gl_cent
            dist_normed = np.linalg.norm(dist_vec)
            sum_radii = gl_rad + pl_rad
            proximal = sum_radii > dist_normed
            proximity_matrix[i, j] = proximal
    proximal_indices = np.transpose(proximity_matrix.nonzero())
    return proximal_indices


def proximity_indices2(bboxes1, bboxes2):
    proximity_matrix = np.zeros([len(bboxes1), len(bboxes2)])
    for i in range(len(bboxes1)):
        bbox1 = bboxes1[i]
        for j, bbox2 in enumerate(bboxes2):
            intersects = bb1_intersects_bb2(bbox1, bbox2)
            proximity_matrix[i, j] = intersects
    proximal_indices = np.transpose(proximity_matrix.nonzero())
    return proximal_indices


def get_1lbl_nbrhoods(labelmap, label, dusting_threshold=5):
    all_labels = get_labels(labelmap)
    remapping = {l: 0 for l in all_labels if l != label}
    labelmap = to_label(labelmap)
    lm_cc = sitk.ChangeLabelLabelMap(labelmap, remapping)
    lm_cc = to_cc(lm_cc)
    fl = sitk.LabelShapeStatisticsImageFilter()
    fl.Execute(lm_cc)
    fl.GetLabels()
    vals = []
    for l in fl.GetLabels():
        rad = fl.GetEquivalentSphericalRadius(l)
        centroid = fl.GetCentroid(l)
        # length = max(fl.GetEquivalentEllipsoidDiameter(l))
        # length = fl.GetFeretDiameter(l)
        vals.append([l, rad, centroid])

    df = pd.DataFrame(vals, columns=["label_cc", "rad", "cent"])
    df["label"] = label
    return lm_cc, df


@astype(22, 0)
def get_all_nbrhoods(labelmap, dusting_threshold=5):
    labels = get_labels(labelmap)
    dfs = []
    for label in labels:
        _, df = get_1lbl_nbrhoods(labelmap, label, dusting_threshold=dusting_threshold)
        dfs.append(df)
    df_final = pd.concat(dfs)
    df_final.reset_index(inplace=True, drop=True)
    return df_final



class ScorerLabelMaps:
    """
    input image, mask, and prediction to compute total dice, lesion-wise dice,  and lesion-wise radiomics (based on mask)
    """

    def __init__(
        self,
        lmg: Union[str, Path],
        lmp: Union[str, Path],
        ignore_labels_g=[],
        ignore_labels_p=[],
        detection_threshold=0.2,
        dusting_threshold=0,
        save_matrices=True,
        results_folder=None,
    ) -> None:
        """
        params_fn: specifies radiomics params
        save_matrices: if true, save full dsc and jac matrices in a separate file
        """

        self.LG = LabelMapGeometry(lmg, ignore_labels_g)
        self.LP = LabelMapGeometry(lmp, ignore_labels_p)
        if not results_folder:
            results_folder = "results"
        store_attr()

    def process(self, debug=False):
        print("Processing ")
        self.dust()
        self.compute_overlap_perlesion()
        self.make_one_to_one_dsc()
        self.compute_overlap_overall()
        self.cont_tables()
        return self.create_df_full()

    @property
    def empty_lm(self):

        if len(self.LG.lengths) == 0 and len(self.LP.lengths) == 0:
            return "both"
        elif (len(self.LG) == 0) ^ (len(self.LP) == 0):
            return "one"

        else:
            return "neither"

    def dust(self):
        # predicted labels <threshold max_dia will be erased
        self.LG.dust(self.dusting_threshold)  # sanity check
        self.LP.dust(self.dusting_threshold)

        self.labs_pred = self.LP.labels
        self.labs_gt = self.LG.labels

    def compute_overlap_overall(self):
        if self.empty_lm == "both":
            self.dsc_overall, self.jac_overall = np.nan, np.nan

        elif self.empty_lm == "one":
            self.dsc_overall, self.jac_overall = 0.0, 0.0
        else:
            self.dsc_overall, self.jac_overall = labels_overlap(
                self.LG.lm_binary, self.LP.lm_binary, 1, 1
            )

    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        # get jaccard and dice

        prox_labels, prox_inds = self.get_neighbr_labels()
        self.LG_ind_cc_pairing = {a[0]: b[0] for a, b in zip(prox_inds, prox_labels)} # lesion ind and matching label 
        self.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}

        self.dsc = np.zeros(
            [max(1, len(self.LP)), max(1, len(self.LG))]
        ).transpose()  # max(1,x) so that an empty matrix is not created
        self.jac = np.copy(self.dsc)
        if self.empty_lm == "neither":
            args = [[self.LG.lm_cc, self.LP.lm_cc, *a] for a in prox_labels]

            d = multiprocess_multiarg(
                labels_overlap, args, 16, False, False, progress_bar=True
            )  # multiprocess i s slow

            # this pairing is a limited number of indices (and corresponding labels) which are in neighbourhoods between LG and LP
            for i, sc in enumerate(d):
                ind_pair = list(prox_inds[i])
                self.dsc[ind_pair[0], ind_pair[1]] = sc[0]
                self.jac[ind_pair[0], ind_pair[1]] = sc[1]

    def make_one_to_one_dsc(self):
        dsc_cp = self.dsc.copy()
        f_labs_matched = np.count_nonzero(dsc_cp, 1)
        m_labs_matched = np.count_nonzero(dsc_cp, 0)
        one_to_many = np.argwhere(f_labs_matched > 1)
        many_to_one = np.argwhere(m_labs_matched > 1)
        if len(many_to_one) > 0:
            dsc_cp[:, many_to_one] = np.apply_along_axis(
                keep_largest, 2, dsc_cp[:, many_to_one]
            )
        if len(one_to_many) > 0:
            dsc_cp[one_to_many, :] = np.apply_along_axis(
                keep_largest, 2, dsc_cp[one_to_many, :]
            )
        self.dsc_single_vals = dsc_cp

    def get_vals_from_indpair(self, indpair):
        return self.dsc_single_vals[indpair[0], indpair[1]]

    def get_neighbr_labels(self):
        prox_inds = proximity_indices(self.LG.nbrhoods, self.LP.nbrhoods)
        nbr1 = self.LG.nbrhoods.iloc[prox_inds[:, 0]]["label_cc"]
        nbr2 = self.LP.nbrhoods.iloc[prox_inds[:, 1]]["label_cc"]
        prox_labels = list(zip(nbr1, nbr2))
        return prox_labels, prox_inds

    def colnames_gen(self, prefix):
        colnames = []
        if len(self.LP.nbrhoods) == 0:
            return ["_".join([prefix, "pred", "label", "dummy"])]
        for ind in self.LP.nbrhoods.index:
            label_cc = self.LP.nbrhoods.loc[ind, "label_cc"]
            label = self.LP.nbrhoods.loc[ind, "label"]
            colname = "_".join([prefix, "pred", "label", str(label), str(label_cc)])
            colnames.append(colname)
        return colnames

    def cont_tables(self):
        """
        param detection_threshold: jaccard b/w lab_mask and lab_ored below this will be counted as a false negative
        """
        if self.LP.nbrhoods.empty:
            self.fp_pred_labels = []
        else:
            tt = self.dsc <= self.detection_threshold
            fp_pred_inds = list(np.where(np.all(tt == True, 0))[0])
            self.fp_pred_labels = list(self.LP.nbrhoods.iloc[fp_pred_inds]["label_cc"])

    def gt_radiomics(self, debug):
        if len(self.labs_gt) == 0:  # or self.do_radiomics == False:
            radiomics = [{"case_id": self.case_id, "label": np.nan}]
        elif self.do_radiomics == False:
            print("No radiomics being done. ")
            radiomics = pd.DataFrame(columns=["case_id", "label"])
            # self.radiomics=[{"case_id":self.case_id, "gt_fn":self.gt_fn ,"label":lab} for lab in self.LG.nbrhoods['label_cc']]

        else:
            radiomics = radiomics_multiprocess(
                self.img,
                self.LG.lm_cc,
                self.labs_gt,
                self.gt_fn,
                self.params_fn,
                debug,
            )
        self.radiomics = pd.DataFrame(radiomics)

    def create_df_full(self):

        pos_inds = np.argwhere(self.dsc_single_vals)
        fk = list(np.arange(pos_inds.shape[0]))
        pos_inds_g = pos_inds[:, 0]
        pos_inds_p = pos_inds[:, 1]

        # gt_all = set(self.LG.nbrhoods.index)
        self.LG.nbrhoods["fk"] = -1  # nan creates a float column
        self.LG.nbrhoods.loc[pos_inds_g, "fk"] = fk
        LG_out = self.LG.nbrhoods.rename(
            mapper=lambda x: "g_" + x if x != "fk" else x, axis=1
        )
        # LG_out = df_rad.merge(
        #     LG_out, right_on="g_label_cc", left_on="label", how="outer"
        # )

        self.LP.nbrhoods["fk"] = -2  # nan creates a float column
        self.LP.nbrhoods.loc[pos_inds_p, "fk"] = fk
        LP_out = self.LP.nbrhoods.rename(
            mapper=lambda x: "p_" + x if x != "fk" else x, axis=1
        )

        dscs = pd.DataFrame([[0, ff] for ff in fk], columns=["dsc", "fk"])
        try:
            dscs["dsc"] = np.apply_along_axis(self.get_vals_from_indpair, 1, pos_inds)
        except:
            pass
        dfs_all = LG_out, LP_out, dscs
        df = reduce(lambda rt, lt: pd.merge(rt, lt, on="fk", how="outer"), dfs_all)
        df["dsc_overall"], df["jac_overall"] = (
            self.dsc_overall,
            self.jac_overall,
        )
        df["g_volume_total"] = self.LG.volume_total
        df["p_volume_total"] = self.LP.volume_total
        # df["case_id"] = df["case_id"].fillna(value=self.case_id)
        df = self.drop_na_rows(df)
        if self.save_matrices == True:
            self.save_overlap_matrices()
        return df

    def drop_na_rows(self, df):
        exc_cols = [
            "case_id",
            "pred_fn",
            "gt_fn",
            "gt_volume_total",
            "pred_volume_total",
        ]
        for i, row in df.iterrows():
            r2 = row.drop(exc_cols)
            if all(r2.isna()):
                df.drop([i], inplace=True)
        return df


class ScorerFiles(ScorerLabelMaps):
    """
    input image, mask, and prediction to compute total dice, lesion-wise dice,  and lesion-wise radiomics (based on mask)
    """

    def __init__(
        self,
        gt_fn: Union[str, Path],
        pred_fn: Union[str, Path],
        img_fn: Union[str, Path] = None,
        case_id=None,
        params_fn=None,
        ignore_labels_gt=[],
        ignore_labels_pred=[],
        detection_threshold=0.2,
        dusting_threshold=0,
        do_radiomics=False,
        save_matrices=True,
        results_folder=None,
    ) -> None:
        """
        params_fn: specifies radiomics params
        save_matrices: if true, save full dsc and jac matrices in a separate file
        """
        if not img_fn:
            assert do_radiomics == False, "To do_radiomics, provide img_fn"
        gt_fn, pred_fn = Path(gt_fn), Path(pred_fn)
        if case_id is None:
            self.case_id = info_from_filename(gt_fn.name, full_caseid=True)["case_id"]
        else:
            self.case_id = case_id
        self.gt, self.pred = [sitk.ReadImage(str(fn)) for fn in [gt_fn, pred_fn]]
        self.img = sitk.ReadImage(str(img_fn)) if img_fn else None

        self.LG = LabelMapGeometry(self.gt, ignore_labels_gt)
        self.LP = LabelMapGeometry(self.pred, ignore_labels_pred)
        if not results_folder:
            results_folder = "results"
        print("Dusting threshold {}".format(dusting_threshold))
        store_attr(but="case_id")

    def process(self, debug=False):
        # try:
        print("Processing {}".format(self.case_id))
        self.dust()
        self.compute_overlap_overall()
        if self.empty_lm == "neither":
            self.compute_overlap_perlesion()
            self.make_one_to_one_dsc()
        self.gt_radiomics(debug)
        self.cont_tables()
        return self.create_df_full()

    # except:
    #     tr()
    # logging.error("Error processing {}".format(self.gt_fn))

    def make_one_to_one_dsc(self):
        dsc_cp = self.dsc.copy()
        f_labs_matched = np.count_nonzero(dsc_cp, 1)
        m_labs_matched = np.count_nonzero(dsc_cp, 0)
        one_to_many = np.argwhere(f_labs_matched > 1)
        many_to_one = np.argwhere(m_labs_matched > 1)
        if len(many_to_one) > 0:
            dsc_cp[:, many_to_one] = np.apply_along_axis(
                keep_largest, 2, dsc_cp[:, many_to_one]
            )
        if len(one_to_many) > 0:
            dsc_cp[one_to_many, :] = np.apply_along_axis(
                keep_largest, 2, dsc_cp[one_to_many, :]
            )
        self.dsc_single_vals = dsc_cp

    def get_vals_from_indpair(self, indpair):
        return self.dsc_single_vals[indpair[0], indpair[1]]

    def get_neighbr_labels(self):
        bboxes_lg = self.LG.nbrhoods["bbox"]
        bboxes_lp = self.LP.nbrhoods["bbox"]
        prox_inds = proximity_indices2(bboxes_lg, bboxes_lp)
        nbr1 = self.LG.nbrhoods.iloc[prox_inds[:, 0]]["label_cc"]
        nbr2 = self.LP.nbrhoods.iloc[prox_inds[:, 1]]["label_cc"]
        prox_labels = list(zip(nbr1, nbr2))
        return prox_labels, prox_inds

    def colnames_gen(self, prefix):
        colnames = []
        if len(self.LP.nbrhoods) == 0:
            return ["_".join([prefix, "pred", "label", "dummy"])]
        for ind in self.LP.nbrhoods.index:
            label_cc = self.LP.nbrhoods.loc[ind, "label_cc"]
            label = self.LP.nbrhoods.loc[ind, "label"]
            colname = "_".join([prefix, "pred", "label", str(label), str(label_cc)])
            colnames.append(colname)
        return colnames

    def cont_tables(self):
        """
        param detection_threshold: jaccard b/w lab_mask and lab_ored below this will be counted as a false negative
        """
        if self.LP.nbrhoods.empty:
            self.fp_pred_labels = []
        else:
            tt = self.dsc <= self.detection_threshold
            fp_pred_inds = list(np.where(np.all(tt == True, 0))[0])
            self.fp_pred_labels = list(self.LP.nbrhoods.iloc[fp_pred_inds]["label_cc"])

    def insert_fks(self, df, dummy_fk, inds, fks):
        repeat_rows = []
        for ind, fk in zip(inds, fks):
            existing_fk = df.loc[ind, "fk"]
            if existing_fk != dummy_fk:
                row = df.loc[ind].copy()
                row["fk"] = fk
                repeat_rows.append(row)
            else:
                df.loc[ind, "fk"] = fk
            # S.LP.nbrhoods.loc[ind_pred, "fk"] = fk

        df = pd.concat([df, pd.DataFrame(repeat_rows)])
        return df

    def create_df_full(self):

        pos_inds = np.argwhere(self.dsc_single_vals)
        fks = list(np.arange(pos_inds.shape[0]))
        pos_inds_gt = pos_inds[:, 0]
        pos_inds_pred = pos_inds[:, 1]

        # gt_all = set(self.LG.nbrhoods.index)
        df_rad = pd.DataFrame(self.radiomics)
        self.LG.nbrhoods = self.insert_fks(self.LG.nbrhoods, -1, pos_inds_gt, fks)
        self.LP.nbrhoods = self.insert_fks(self.LP.nbrhoods, -2, pos_inds_pred, fks)

        LG_out1 = self.LG.nbrhoods.rename(
            mapper=lambda x: "gt_" + x if x != "fk" else x, axis=1
        )
        LG_out2 = df_rad.merge(
            LG_out1, right_on="gt_label_cc", left_on="label", how="outer"
        )

        LP_out = self.LP.nbrhoods.rename(
            mapper=lambda x: "pred_" + x if x != "fk" else x, axis=1
        )

        dscs = pd.DataFrame([[0, ff] for ff in fks], columns=["dsc", "fk"])
        try:
            dscs["dsc"] = np.apply_along_axis(self.get_vals_from_indpair, 1, pos_inds)
        except:
            pass
        dfs_all = LG_out2, LP_out, dscs
        df = reduce(lambda rt, lt: pd.merge(rt, lt, on="fk", how="outer"), dfs_all)
        df["pred_fn"] = self.pred_fn
        df["gt_fn"] = self.gt_fn
        df["dsc_overall"], df["jac_overall"] = (
            self.dsc_overall,
            self.jac_overall,
        )
        df["gt_volume_total"] = self.LG.volume_total
        df["pred_volume_total"] = self.LP.volume_total
        df["case_id"] = df["case_id"].fillna(value=self.case_id)

        df = self.cleanup(df)
        if self.save_matrices == True:
            self.save_overlap_matrices()
        return df

    def cleanup(self, df):
        if self.LP.is_empty() == True:  # -2 fk is used to count false positives by R.
            locs = df.fk != -2
            df = df.loc[locs]
        df = self.drop_na_rows(df)
        redundant_cols = ["gt_bbox", "gt_rad", "pred_bbox", "pred_rad"]
        redundant_cols = set(redundant_cols).intersection(df.columns)
        df.drop(redundant_cols, axis=1, inplace=True)
        return df

    def drop_na_rows(self, df):
        exc_cols = [
            "case_id",
            "pred_fn",
            "gt_fn",
            "gt_volume_total",
            "pred_volume_total",
        ]
        rel_cols = df.columns.intersection(exc_cols)
        drop_inds = []
        for i, row in df.iterrows():
            r2 = row.drop(exc_cols)
            if all(r2.isna()):
                drop_inds.append(i)
        drops = df.index.isin(drop_inds)
        df = df[~drops]
        return df

    def save_overlap_matrices(self):
        dsc_labels = self.colnames_gen("dsc")
        jac_labels = self.colnames_gen("jac")
        row_labels = self.LG.nbrhoods["label_cc"].tolist()
        dsc_df = pd.DataFrame(data=self.dsc, columns=dsc_labels, index=row_labels)
        jac_df = pd.DataFrame(data=self.jac, columns=jac_labels, index=row_labels)
        dsc_df.to_csv(
            self.results_folder / ("{}_dsc.csv".format(self.case_id)),
        )

        jac_df.to_csv(
            self.results_folder / ("{}_jac.csv".format(self.case_id)),
        )

    @property
    def results_folder(self):
        """The results_folder property."""
        return self._results_folder

    @results_folder.setter
    def results_folder(self, value):
        self._results_folder = Path(value)
        maybe_makedirs(self._results_folder)


class ScorerAdvanced(ScorerFiles):
    def corresponding_inds(self, ref_inds, dsc_axis):
        # dsc_axis= 0 if gt_inds provided, 1 if pred_inds provided
        dsc_gps = []
        for ind in ref_inds:
            if hasattr(ind, "item"):
                ind = ind.item()
            if dsc_axis == 0:
                gp = np.nonzero(S.dsc[ind, :])
            else:
                gp = np.nonzero(S.dsc[:, ind])
            gp = set(gp[0])
            dsc_gps.append(gp)
        return dsc_gps

    def dsc_from_cc_pair(self, cc_pair):
        cc = list(cc_pair)
        for c in cc:
            lab = re.findall(r"\d+", c)[0]
            lab = int(lab)
            if "gt" in c:
                inx = key_from_value(S.LG_ind_cc_pairing, lab)[0]
                row = inx
            else:
                inx = key_from_value(S.LP_ind_cc_pairing, lab)[0]
                col = inx
        dsc = self.dsc[row, col]
        return dsc

    def dsc_gp_remapping(self, dsc_gps):
        remapping = {}
        dest_labels = []
        for gp in dsc_gps:
            gp = inds_to_labels(gp)
            main_lab = gp[0]
            dest_labels.append(main_lab)
            maps = {lab: int(main_lab) for lab in gp}
            remapping.update(maps)
        return remapping, dest_labels

    def _dsc_multilabel(self, prox_labels):
        args = [[self.LG.lm_cc, self.LP.lm_cc, *a] for a in prox_labels]
        d = multiprocess_multiarg(
            labels_overlap, args, 8, False, False, progress_bar=True
        )  # multiprocess i s slow
        return d

    def recompute_overlap_perlesion(self):

        r = self.dsc.shape[0]
        s = self.dsc.shape[1]
        gt_labs = self.LG.nbrhoods["label_cc"].tolist()
        gt_labs = "gt_lab_"+ self.LG.nbrhoods["label_cc"].astype(str)
        gt_labs = []
        pred_labs ="pred_lab_"+  self.LP.nbrhoods["label_cc"].astype(str)
        G = nx.Graph()
        G.add_nodes_from(gt_labs, bpartite=0)
        G.add_nodes_from(pred_labs, bpartite=1)
        edge_sets = []
        for row in range(r):
            if row in self.LG_ind_cc_pairing.keys():
                gt_lab = "gt_lab_"+str(self.LG_ind_cc_pairing[row])
                edges = np.argwhere(self.dsc[row, :]).flatten().tolist()
                pred_labs= ["pred_lab_" + str(self.LP_ind_cc_pairing[ind]) for ind in edges]
                gt_ind_2_preds = [(gt_lab, pred_lab) for pred_lab in pred_labs]
                edge_sets.extend(gt_ind_2_preds)
            else:
                pass # This index LG has no overlapping pred


        G.add_edges_from(edge_sets)
        con = nx.connected_components(G)
        ccs = list(con)

        fk_gen = fk_generator(start=1)
        gt_remaps = {}
        pred_remaps = {}
        fks = []
        self.df2 = []
        for cc in ccs:
            fk = next(fk_gen)
            fks.append(fk)
            dici = {"fk": fk}
            gt_labels = []
            pred_labels = []
            recompute_dsc = False
            dsc = float("nan")
            if len(cc) > 2:
                recompute_dsc = True
            elif len(cc) == 2:
                dsc = self.dsc_from_cc_pair(cc)
            while cc:
                label = cc.pop()
                indx = re.findall(r"\d+", label)[0]
                indx = int(indx)
                if "gt" in label:
                    gt_labels.append(indx)
                    gt_remaps.update({indx: fk})
                else:
                    pred_labels.append(indx)
                    pred_remaps.update({indx: fk})
            labels = {
                "pred_label_cc": pred_labels,
                "gt_label_cc": gt_labels,
                "recompute_dsc": recompute_dsc,
                "dsc": dsc,
            }
            dici.update(labels)
            self.df2.append(dici)
        self.df2 = pd.DataFrame(self.df2)
        self.LG._relabel(gt_remaps)
        self.LP._relabel(pred_remaps)

        redsc_labels = self.df2.loc[self.df2["recompute_dsc"] == True, "fk"].tolist()
        prox_labels = [[a, a] for a in redsc_labels]

        dsc_jac_multi = self._dsc_multilabel(prox_labels)

        self.dsc_multi = [a[0] for a in dsc_jac_multi]
        self.df2.loc[self.df2["fk"].isin(redsc_labels), "dsc"] = self.dsc_multi

    def insert_fks(self, nbrhoods, label_key):
        colnames = ["label", "cent", "length", "volume", "label_cc"]
        dfs = []
        for i in range(len(self.df2)):
            cluster = self.df2.iloc[i]
            labels = cluster[label_key]
            fk = cluster["fk"]
            row = nbrhoods.loc[nbrhoods["label_cc"].isin(cluster[label_key])]
            # if len(labels)>1:
            #     label_dom = [row['label'].max()]
            #     cent = [row['cent'].tolist()[0]]
            #     length = [row['length'].sum()]
            #     volume = [row['volume'].sum()]
            #     df_dict = {'label':label_dom, 'cent':cent,'length':length, 'volume':volume,'label_cc':fk ,'fk':fk}
            #     df_mini = pd.DataFrame(df_dict)
            #     dfs.append(df_mini)
            # elif len(labels)==1:
            df_mini = row[colnames]
            df_mini = df_mini.assign(fk=fk)
            dfs.append(df_mini)
        df_final = pd.concat(dfs, axis=0)
        return df_final

    def insert_dsc_fks(self):
        self.LG.nbrhoods["dsc"] = float("nan")
        if self.empty_lm != "neither":
            self.LG.nbrhoods["fk"] = -1
            self.LP.nbrhoods["fk"] = -2
        else:
            self.LG.nbrhoods = self.insert_fks(self.LG.nbrhoods, "gt_label_cc")
            self.LP.nbrhoods = self.insert_fks(self.LP.nbrhoods, "pred_label_cc")

    def create_df_full(self):

        df_rad = pd.DataFrame(self.radiomics)
        LG_out1 = self.LG.nbrhoods.rename(
            mapper=lambda x: "gt_" + x if not x in ["dsc", "fk"] else x, axis=1
        )
        LG_out2 = df_rad.merge(
            LG_out1, right_on="gt_label_cc", left_on="label", how="outer"
        )

        LP_out = self.LP.nbrhoods.rename(
            mapper=lambda x: "pred_" + x if x != "fk" else x, axis=1
        )
        if hasattr(self, "df2"):
            dfs_all = [LG_out2, LP_out, self.df2[["fk", "dsc"]]]
        else:
            dfs_all = [LG_out2, LP_out]
        df = reduce(lambda rt, lt: pd.merge(rt, lt, on="fk", how="outer"), dfs_all)

        df["pred_fn"] = self.pred_fn
        df["gt_fn"] = self.gt_fn
        df["dsc_overall"], df["jac_overall"] = (
            self.dsc_overall,
            self.jac_overall,
        )
        df["gt_volume_total"] = self.LG.volume_total
        df["pred_volume_total"] = self.LP.volume_total
        df["case_id"] = df["case_id"].fillna(value=self.case_id)

        self.df_final = self.cleanup(df)
        return self.df_final

    def process(self, debug=False):
        try:
            print("Processing {}".format(self.case_id))
            self.dust()
            self.compute_overlap_overall()
            if self.empty_lm == "neither":
                self.compute_overlap_perlesion()
                self.recompute_overlap_perlesion()
            self.gt_radiomics(debug)
            self.insert_dsc_fks()
            return self.create_df_full()
        except:
            logging.error("Error processing {}".format(self.gt_fn))
            raise ValueError


class BatchScorer:
    def __init__(
        self,
        gt_fns: Union[Path, list],
        preds_fldr: Path,
        ignore_labels_gt: list,
        ignore_labels_pred: list,
        imgs_fldr: Path = None,
        partial_df: pd.DataFrame = None,
        exclude_fns=[],
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=3,
        debug=False,
    ):

        if isinstance(gt_fns, Path):
            self.gt_fns = list(gt_fns.glob("*"))
        if len(exclude_fns) > 0:
            self.gt_fns = [fn for fn in self.gt_fns if fn not in exclude_fns]

        if output_fldr is None:
            output_fldr = preds_fldr / ("results")
        store_attr(
            "partial_df,output_fldr,do_radiomics,debug,ignore_labels_gt, ignore_labels_pred,dusting_threshold,preds_fldr,imgs_fldr"
        )
        gt_fns = self.filter_gt_fns(gt_fns, partial_df, exclude_fns)
        self.file_dicts = self.match_filenames(gt_fns)

    def process(self):
        self.dfs = []
        for fn_dict in self.file_dicts:
            gt_fn, pred_fn, img_fn = fn_dict.values()
            print("processing {}".format(gt_fn))
            S = ScorerAdvanced(
                gt_fn=gt_fn,
                img_fn=img_fn,
                pred_fn=pred_fn,
                ignore_labels_gt=self.ignore_labels_gt,
                ignore_labels_pred=self.ignore_labels_pred,
                case_id=None,
                save_matrices=False,
                do_radiomics=self.do_radiomics,
                dusting_threshold=self.dusting_threshold,
            )
            df = S.process(debug=self.debug)
            self.dfs.append(df)
            self.store_tmp_df()
        df_final = self.finalise_df(self.dfs)
        # self.df = pd.concat(dfs, axis=0)
        print("Saving results to {}".format(self.output_fn))
        df_final.to_excel(self.output_fn, index=False)
        return df_final

    def store_tmp_df(self):
        dfs_tmp = pd.concat(self.dfs)
        dfs_tmp.to_csv(self.output_fldr / "tmp.csv")

    def finalise_df(self, dfs):
        if self.partial_df is not None:
            dfs += [self.partial_df]
        df_final = pd.concat(dfs, axis=0)
        return df_final

    def match_filenames(self, gt_fns):
        pred_fns = list(self.preds_fldr.glob("*"))
        if self.imgs_fldr:
            img_fns = list(imgs_fldr.glob("*.*"))
        else:
            img_fns = None
        file_dicts = []
        for gt_fn in gt_fns:
            case_id = info_from_filename(gt_fn.name, full_caseid=True)["case_id"]
            pred_fn = [fn for fn in pred_fns if case_id in fn.name]
            if len(pred_fn) != 1:
                tr()
            else:
                pred_fn = pred_fn[0]
            fn_dict = {"gt_fn": gt_fn, "pred_fn": pred_fn, "img_fn": None}
            if img_fns:
                img_fn = [
                    fn for fn in img_fns if match_filenames(gt_fn.name, fn.name) == True
                ]
                if len(img_fn) != 1:
                    tr()
                else:
                    img_fn = img_fn[0]
                fn_dict["img_fn"] = img_fn
            file_dicts.append(fn_dict)
        return file_dicts

    def filter_gt_fns(self, gt_fns, partial_df, exclude_fns):
        print("Total gt files: {}".format(len(gt_fns)))
        print("Excluded files: {}".format(len(exclude_fns)))
        cid_done = []
        exclude_fns = []
        if partial_df is not None:
            cid_done = list(partial_df["case_id"].unique())
        if len(exclude_fns) > 0:
            cid_done.append(
                [info_from_filename(fn, True)["case_id"] for fn in exclude_fns]
            )

        fns_pending = [
            fn
            for fn in gt_fns
            if info_from_filename(fn.name, True)["case_id"] not in cid_done
        ]
        print(
            "After filtering already processed files, files remaining: {}".format(
                len(fns_pending)
            )
        )

        return fns_pending

    @property
    def output_fn(self):
        maybe_makedirs(self.output_fldr)
        output_fn = self.output_fldr.name + "_thresh{0}mm_results.xlsx".format(
            self.dusting_threshold
        )
        output_fn = self.output_fldr / output_fn
        return output_fn


class BatchScorer2(BatchScorer):

    def __init__(
        self,
        output_suffix,
        gt_fns: Union[Path, list],
        preds_fldr: Path,
        ignore_labels_gt: list,
        ignore_labels_pred: list,
        imgs_fldr: Path = None,
        partial_df: pd.DataFrame = None,
        exclude_fns=[],
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=3,
        debug=False,
    ):
        self.output_suffix = output_suffix
        assert isinstance(
            output_suffix, int
        ), "Need integer output_suffix, got {}".format(output_suffix)
        super().__init__(
            gt_fns,
            preds_fldr,
            ignore_labels_gt,
            ignore_labels_pred,
            imgs_fldr,
            partial_df,
            exclude_fns,
            output_fldr,
            do_radiomics,
            dusting_threshold,
            debug,
        )

    def store_tmp_df(self):
        dfs_tmp = pd.concat(self.dfs)
        dfs_tmp.to_csv(self.output_fldr / "tmp{}.csv".format(self.output_suffix))

    @property
    def output_fn(self):
        if self.output_fldr is None:
            self.output_fldr = self.preds_fldr / ("results")
        maybe_makedirs(self.output_fldr)
        output_fn = self.output_fldr.name + "_thresh{0}mm_results{1}.xlsx".format(
            self.dusting_threshold, self.output_suffix
        )
        output_fn = self.output_fldr / output_fn
        return output_fn


@ray.remote(num_cpus=4)
class BatchScorerRay:
    def __init__(self, actor_id):
        self.actor_id = actor_id

    def process(
        self,
        gt_fns: Union[Path, list],
        preds_fldr: Path,
        ignore_labels_gt: list,
        ignore_labels_pred: list,
        imgs_fldr: Path = None,
        partial_df: pd.DataFrame = None,
        exclude_fns=[],
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
    ):
        print("process {} ".format(self.actor_id))
        self.B = BatchScorer2(
            output_suffix=self.actor_id,
            gt_fns=gt_fns,
            preds_fldr=preds_fldr,
            ignore_labels_gt=ignore_labels_gt,
            ignore_labels_pred=ignore_labels_pred,
            imgs_fldr=imgs_fldr,
            partial_df=partial_df,
            exclude_fns=exclude_fns,
            do_radiomics=do_radiomics,
            dusting_threshold=dusting_threshold,
            debug=debug,
        )
        return self.B.process()


#
# B2 = BatchScorer2(1,*argi)
# df = B2.process()
# class BatchActor():
#     def __init__(self,actor_id):self.actor_id =actor_id
#
#     def process(self,
#         gt_fns,
#         preds_fldr,
#         ignore_labels_gt=[],
#         ignore_labels_pred=[],
#         imgs_fldr=None,
#         partial_df=None,
#         debug=False,
#         do_radiomics=False,
#         dusting_threshold=0,
#                 output_fldr=None):
#         B = BatchScorerRay(gt_fns=gt_fns, preds_fldr=preds_fldr, partial_df=partial_df, imgs_fldr=imgs_fldr,ignore_labels_gt=ignore_labels_gt,ignore_labels_pred=ignore_labels_pred,do_radiomics=do_radiomics,dusting_threshold=dusting_threshold, debug=debug)


if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc")
    preds_nnunet_fldr = Path("/s/datasets_bkp/ark/")
    gt_fldr = Path("/s/xnat_shadow/crc/lms")
    gt_fns = list(gt_fldr.glob("*"))
    gt_fns = [fn for fn in gt_fns if is_sitk_file(fn)]

    ub_df_fn = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh1mm.xlsx"
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    results_df = pd.read_excel(ub_df_fn)

    partial_df = None
    partial_df = results_df
# %%
# SECTION:-------------------- Batchscorer Ray--------------------------------------------------------------------------------------

    n_lists = 10
    # gt_fns = gt_fns[:64]
    fpl = int(len(gt_fns) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])
    chunks = list(il.starmap(slice_list, zip([gt_fns] * n_lists, inds)))
    dusting_threshold=0

    args_nnunet = [
        [
            chunk,
            preds_nnunet_fldr,
            [1],
            [1],
            None,
            partial_df,
            [],
            None,
            False,
            dusting_threshold,
            False,
        ]
        for chunk in chunks
    ]
# %%
    
    args = [
        [chunk, preds_fldr, [1], [1], None, partial_df, [], None, False, dusting_threshold, False]
        for chunk in chunks
    ]


# %%

    actors = [BatchScorerRay.remote(id) for id in range(n_lists)]
# %%
    results_nnunet = ray.get(
        [c.process.remote(*a) for c, a in zip(actors, args_nnunet)]
    )
    df_nnunet = pd.concat(results_nnunet)
    df_fn_nnunet = preds_nnunet_fldr / ("results/results.xlsx")
    df_nnunet.to_excel(df_fn_nnunet, index=False)
# %%
    results = ray.get([c.process.remote(*a) for c, a in zip(actors, args)])
# %%
    df = pd.concat(results)
    df.to_excel(ub_df_fn, index=False)
# %%
    df = pd.read_excel(ub_df_fn)
    bb = set(df.case_id)
    bb.difference(aa)
# %%
# SECTION:-------------------- Batchscorer single --------------------------------------------------------------------------------------

    do_radiomics = False
    threshold = 0

# %%
    # partial_df = None
    B = BatchScorer(
        gt_fns,
        preds_fldr=preds_fldr,
        ignore_labels_gt=[],
        ignore_labels_pred=[1],
        imgs_fldr=None,
        partial_df=partial_df,
        debug=False,
        do_radiomics=False,
        dusting_threshold=threshold,
        output_fldr=Path("/home/ub/code/label_analysis/results"),
    )  # ,output_fldr=Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/results"))
# %%
    B.process()
# %%
    arkf = list(preds_nnunet_fldr.glob("*"))
    fns = [fn.name for fn in arkf]

    outfldr = Path("/s/datasets_bkp/ark/imgs_pending")
    imgs_fldr = Path("/s/xnat_shadow/crc/images")
    im = list(imgs_fldr.glob("*"))
    ims = [fn.name for fn in im]
    aa = list(set(ims).difference(fns))
    import shutil

# %%
    for fn in aa:
        in_f = imgs_fldr / fn
        out_f = outfldr / fn
        shutil.copy(in_f, out_f)

# %%

# %%
# SECTION:-------------------- FILE SCOorer (ScorerAdvanced)--------------------------------------------------------------------------------------

    cid = "crc_CRC211"

    gt_fn = [fn for fn in gt_fns if cid in fn.name][0]
    pred_ub = find_matching_fn(gt_fn, preds_fldr, True)

    S = ScorerAdvanced(
        gt_fn, pred_ub, ignore_labels_gt=[], ignore_labels_pred=[1]
    )
# %%
# %%
    df = S.process()
    df.to_csv("crc211.csv")
    sitk.WriteImage(S.LP.lm_cc,"pred_cc.nii.gz")
    sitk.WriteImage(S.LG.lm_cc,"gt_cc.nii.gz")
# %%
# %%
#SECTION:-------------------- SCORER SLICEROUTPUT--------------------------------------------------------------------------------------

    # gt_fn2 = "/home/ub/code/label_analysis/gt_cc_slicer.nrrd"
    # pred_fn2= "/home/ub/code/label_analysis/pred_cc_slicer.nrrd"
    #
    # S = ScorerAdvanced(
    #     gt_fn2, pred_fn2, ignore_labels_gt=[], ignore_labels_pred=[]
    # )
# %%
    # lg_ar = sitk.GetArrayFromImage(S.LG.lm_cc)
    # lp_ar = sitk.GetArrayFromImage(S.LP.lm_cc)
    # ImageMaskViewer([lg_ar, lp_ar], 'mm')
# %%
    df = S.process()
    df.to_csv("crc211.csv")
    sitk.WriteImage(S.LP.lm_cc,"pred_cc.nii.gz")
    sitk.WriteImage(S.LG.lm_cc,"gt_cc.nii.gz")

# SECTION:-------------------- START HERE fk 11 has two gt_label_cc--------------------------------------------------------------------------------------

    fn1 = "/home/ub/code/label_analysis/pred_19.nrrd"
    fn2 = "/home/ub/code/label_analysis/gt_22.nrrd"
    S = ScorerAdvanced(
        fn1, fn2, ignore_labels_gt=[], ignore_labels_pred=[], case_id="test"
    )
# %%
    # df1 = S.process()
    # S = ScorerAdvanced(gt_fn, pred_ub,ignore_labels_gt=[1],ignore_labels_pred=[1])
    # df = S.process()

# %%
    S.dusting_threshold
    deb = False
    print("Processing {}".format(S.case_id))
    S.dust()
    S.compute_overlap_overall()
# %%
    if S.empty_lm == "neither":
        S.compute_overlap_perlesion()
        S.recompute_overlap_perlesion()
    S.gt_radiomics(deb)
# %%
    S.insert_dsc_fks()
    df_f = S.create_df_full()
# %%
    r = S.dsc.shape[0]
    s = S.dsc.shape[1]
    gt_labs = S.LG.nbrhoods["label_cc"].tolist()
    gt_labs = "gt_lab_"+ S.LG.nbrhoods["label_cc"].astype(str)
    gt_labs = []
    pred_labs ="pred_lab_"+  S.LP.nbrhoods["label_cc"].astype(str)
    G = nx.Graph()
    G.add_nodes_from(gt_labs, bpartite=0)
    G.add_nodes_from(pred_labs, bpartite=1)
    edge_sets = []
    for row in range(r):
        if row in S.LG_ind_cc_pairing.keys():
            gt_lab = "gt_lab_"+str(S.LG_ind_cc_pairing[row])
            edges = np.argwhere(S.dsc[row, :]).flatten().tolist()
            pred_labs= ["pred_lab_" + str(S.LP_ind_cc_pairing[ind]) for ind in edges]
            gt_ind_2_preds = [(gt_lab, pred_lab) for pred_lab in pred_labs]
            edge_sets.extend(gt_ind_2_preds)
        else:
            pass # This index LG has no overlapping pred


    G.add_edges_from(edge_sets)
    con = nx.connected_components(G)
    ccs = list(con)

    fk_gen = fk_generator(start=1)
    gt_remaps = {}
    pred_remaps = {}
    fks = []
    S.df2 = []
    for cc in ccs:
        fk = next(fk_gen)
        fks.append(fk)
        dici = {"fk": fk}
        gt_labels = []
        pred_labels = []
        recompute_dsc = False
        dsc = float("nan")
        if len(cc) > 2:
            recompute_dsc = True
        elif len(cc) == 2:
            dsc = S.dsc_from_cc_pair(cc)
        while cc:
            label = cc.pop()
            indx = re.findall(r"\d+", label)[0]
            indx = int(indx)
            if "gt" in label:
                gt_labels.append(indx)
                gt_remaps.update({indx: fk})
            else:
                pred_labels.append(indx)
                pred_remaps.update({indx: fk})
        labels = {
            "pred_label_cc": pred_labels,
            "gt_label_cc": gt_labels,
            "recompute_dsc": recompute_dsc,
            "dsc": dsc,
        }
        dici.update(labels)
        S.df2.append(dici)


 
# %%
    prox_labels, prox_inds = S.get_neighbr_labels()
    S.LG_ind_cc_pairing = {a[0]: b[0] for a, b in zip(prox_inds, prox_labels)}
    S.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}

    S.dsc = np.zeros(
        [max(1, len(S.LP)), max(1, len(S.LG))]
    ).transpose()  # max(1,x) so that an empty matrix is not created
    S.jac = np.copy(S.dsc)
    if S.empty_lm == "neither":
        args = [[S.LG.lm_cc, S.LP.lm_cc, *a] for a in prox_labels]

        d = multiprocess_multiarg(
            labels_overlap, args, 16, False, False, progress_bar=True
        )  # multiprocess i s slow

        # this pairing is a limited number of indices (and corresponding labels) which are in neighbourhoods between LG and LP
        for i, sc in enumerate(d):
            ind_pair = list(prox_inds[i])
            S.dsc[ind_pair[0], ind_pair[1]] = sc[0]
            S.jac[ind_pair[0], ind_pair[1]] = sc[1]

# %%
    #
    S.LG.nbrhoods["dsc"] = float("nan")
    if S.empty_lm != "neither":
        S.LG.nbrhoods["fk"] = -1
        S.LP.nbrhoods["fk"] = -2
    else:
        S.LG.nbrhoods = S.insert_fks(S.LG.nbrhoods, "gt_label_cc")
        S.LP.nbrhoods = S.insert_fks(S.LP.nbrhoods, "pred_label_cc")

# %%

    nbg = S.insert_fks(S.LG.nbrhoods, "gt_label_cc")
    nbp = S.insert_fks(S.LP.nbrhoods, "pred_label_cc")
    df_f2 = S.create_df_full()
# %%
    nbrhoods = S.LG.nbrhoods.copy()
    label_key = "gt_label_cc"

    colnames = ["label", "cent", "length", "volume", "label_cc"]
    dfs = []
    for i in range(len(S.df2)):
        cluster = S.df2.iloc[i]
        labels = cluster[label_key]
        fk = cluster["fk"]
        row = nbrhoods.loc[nbrhoods["label_cc"].isin(cluster[label_key])]
        # if len(labels)>1:
        #     tr()
        #     label_dom = [row['label'].max()]
        #     cent = [row['cent'].tolist()[0]]
        #     length = [row['length'].sum()]
        #     volume = [row['volume'].sum()]
        #     df_dict = {'label':label_dom, 'cent':cent,'length':length, 'volume':volume,'label_cc':fk ,'fk':fk}
        #     df_mini = pd.DataFrame(df_dict)
        #     dfs.append(df_mini)
        # elif len(labels)==1:
        df_mini = row[colnames]
        df_mini = df_mini.assign(fk=fk)
        dfs.append(df_mini)

    df_final = pd.concat(dfs, axis=0)

# %%
# SECTION:-------------------- ROUGHT--------------------------------------------------------------------------------------

# %%

# %%
# SECTION:-------------------- SORT out --------------------------------------------------------------------------------------

# %%
    out_fldr_missed = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/missed_subcm/"
    )
    out_fldr_missed_binary = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/missed_subcm_binary/"
    )
    out_fldr_detected = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/detected_subcm/"
    )
    out_fldr_detected_binary = Path(
        "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/detected_subcm_binary/"
    )

# %%
    for lm_fn in gt_fns:
        cid = info_from_filename(lm_fn.name, full_caseid=True)["case_id"]
        sub_df = results_df[results_df["case_id"] == cid]
        sub_df = sub_df[sub_df["fk"] > 0]
        missed = sub_df[sub_df["dsc"].isna()]
        missed = missed[missed["gt_length"] <= 10]
        if len(missed) > 0:

            lm = sitk.ReadImage(str(lm_fn))
            L = LabelMapGeometry(lm)
            cents = missed["gt_cent"].tolist()
            excluded = L.nbrhoods[L.nbrhoods["length"] > 10]
            excluded = excluded["label_cc"].tolist()
            cents = [ast.literal_eval(c) for c in cents]
            remaps = {x: 0 for x in excluded}
            L.lm_cc = relabel(L.lm_cc, remaps)

            missed_nbr = L.nbrhoods[L.nbrhoods["cent"].isin(cents)]
            missed_labs = missed_nbr["label_cc"].tolist()
            other_labs = L.nbrhoods[~L.nbrhoods["label_cc"].isin(missed_labs)]
            other_labs = other_labs["label_cc"].tolist()
            remapping_missed = {x: 0 for x in other_labs}
            remapping_detected = {x: 0 for x in missed_labs}
# %%
            lm_missed = relabel(L.lm_cc, remapping_missed)

            sitk.WriteImage(lm_missed, str(out_fldr_missed / lm_fn.name))

            lm_missed_binary = to_binary(lm_missed)
            sitk.WriteImage(lm_missed_binary, str(out_fldr_missed_binary / lm_fn.name))
            lm_detected = relabel(L.lm_cc, remapping_detected)
            sitk.WriteImage(lm_detected, str(out_fldr_detected / lm_fn.name))
            lm_detected_binary = to_binary(lm_detected)
            sitk.WriteImage(
                lm_detected_binary, str(out_fldr_detected_binary / lm_fn.name)
            )
# %%

    view_sitk(L.lm_cc, L.lm_cc)

    gt_fns.sort(key=os.path.getmtime, reverse=True)
    files_pending = [fn for fn in gt_fns if test_modified(fn, 5) == True]
    cids = [info_from_filename(fn.name)["case_id"] for fn in files_pending]
    cids = ["crc_" + cid for cid in cids]

    done = results_df.loc[~results_df["case_id"].isin(cids)]
    # partial_df = results_df.loc[done]
    partial_df = None
    partial_df = done
    # fns_pending = [fn for fn in gt_fns if info_from_filename(fn.name)['case_id'] not in cid_done]
    # cid_done = set(partial_df['case_id'].values)
    # crc1 = "CRC003"
    # crc2 = "CRC275"
    # gt_fns2 = [find_file(crc1,gt_fns), find_file(crc2,gt_fns)]


# %%
# %%
# %%
# %%
