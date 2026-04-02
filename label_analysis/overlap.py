# %%
import sys
from functools import reduce

import networkx as nx
from utilz.cprint import cprint
from utilz.dictopts import key_from_value

from label_analysis.geometry import (LabelMapGeometry,
                                     load_labelmap_like_to_sitk)
from label_analysis.geometry_itk import LabelMapGeometryITK
from label_analysis.geometry_pt import LabelMapGeometryPT

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from utilz.fileio import load_json, maybe_makedirs
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.stringz import info_from_filename, match_filenames

from label_analysis.helpers import *

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

pd.set_option("display.width", 200)           # total line width before
pd.set_option("display.max_columns", None)    # show all columns
pd.set_option("display.max_colwidth", 25)     # truncate long cell contents

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
    df["label_org"] = label
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
        self.lmg = lmg
        self.lmp = lmp
        self.ignore_labels_g = ignore_labels_g
        self.ignore_labels_p = ignore_labels_p
        self.detection_threshold = detection_threshold
        self.dusting_threshold = dusting_threshold
        self.save_matrices = save_matrices
        self.results_folder = results_folder

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
        print("Dusting GT")
        self.LG.dust(self.dusting_threshold)  # sanity check
        print("-" * 20)
        print("Dusting Pred")
        self.LP.dust(self.dusting_threshold)
        print("-" * 20)


    def compute_overlap_overall(self):
        if self.empty_lm == "both":
            self.dsc_overall, self.jac_overall = np.nan, np.nan

        elif self.empty_lm == "one":
            self.dsc_overall, self.jac_overall = 0.0, 0.0
        else:
            self.dsc_overall, self.jac_overall = labels_overlap(
                self.LG.li_binary, self.LP.li_binary, 1, 1
            )

    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        # get jaccard and dice

        prox_labels, prox_inds = self.get_neighbr_labels()
        self.LG_ind_cc_pairing = {
            a[0]: b[0] for a, b in zip(prox_inds, prox_labels)
        }  # lesion ind and matching label
        self.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}

        self.dsc = np.zeros(
            [max(1, len(self.LP)), max(1, len(self.LG))]
        ).transpose()  # max(1,x) so that an empty matrix is not created
        self.jac = np.copy(self.dsc)
        if self.empty_lm == "neither":
            args = [[self.LG.li_cc_sitk, self.LP.li_cc_sitk, *a] for a in prox_labels]

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
            label_org = self.LP.nbrhoods.loc[ind, "label_org"]
            colname = "_".join([prefix, "pred", "label", str(label_org), str(label_cc)])
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
            from label_analysis.radiomics_setup import radiomics_multiprocess

            radiomics = radiomics_multiprocess(
                self.img,
                self.LG.li_cc_sitk,
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

    @property
    def labs_gt(self):
        return self.LG.labels
    @property
    def labs_pred(self):
        return self.LP.labels

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
        self.gt_fn = gt_fn
        self.pred_fn = pred_fn
        self.img_fn = img_fn
        self.params_fn = params_fn
        self.ignore_labels_gt = ignore_labels_gt
        self.ignore_labels_pred = ignore_labels_pred
        self.detection_threshold = detection_threshold
        self.dusting_threshold = dusting_threshold
        self.do_radiomics = do_radiomics
        self.save_matrices = save_matrices
        self.results_folder = results_folder

    def process(self, debug=False):
        try:
            print("Processing {}".format(self.case_id))
            self.dust()
            self.compute_overlap_overall()
            if self.empty_lm == "neither":
                self.compute_overlap_perlesion()
                self.make_one_to_one_dsc()
            self.gt_radiomics(debug)
            self.cont_tables()
            return self.create_df_full()

        except Exception as e:
            msg = "Error processing case_id={} gt_fn={} pred_fn={}".format(
                self.case_id, self.gt_fn, self.pred_fn
            )
            logging.exception(msg)
            raise RuntimeError(msg) from e

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
            label_org = self.LP.nbrhoods.loc[ind, "label_org"]
            colname = "_".join([prefix, "pred", "label", str(label_org), str(label_cc)])
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
        df.columns.intersection(exc_cols)
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
                gp = np.nonzero(self.dsc[ind, :])
            else:
                gp = np.nonzero(self.dsc[:, ind])
            gp = set(gp[0])
            dsc_gps.append(gp)
        return dsc_gps

    def dsc_from_cc_pair(self, cc_pair):
        cc = list(cc_pair)
        for c in cc:
            lab = re.findall(r"\d+", c)[0]
            lab = int(lab)
            if "gt" in c:
                inx = key_from_value(self.LG_ind_cc_pairing, lab)[0]
                row = inx
            else:
                inx = key_from_value(self.LP_ind_cc_pairing, lab)[0]
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
        args = [[self.LG.li_cc_sitk, self.LP.li_cc_sitk, *a] for a in prox_labels]
        d = multiprocess_multiarg(
            labels_overlap, args, 8, False, False, progress_bar=True
        )  # multiprocess i s slow
        return d

    def recompute_overlap_perlesion(self):
        r = self.dsc.shape[0]
        s = self.dsc.shape[1]
        gt_labs = self.LG.nbrhoods["label_cc"].tolist()
        gt_labs = "gt_lab_" + self.LG.nbrhoods["label_cc"].astype(str)
        gt_labs = []
        pred_labs = "pred_lab_" + self.LP.nbrhoods["label_cc"].astype(str)
        G = nx.Graph()
        G.add_nodes_from(gt_labs, bpartite=0)
        G.add_nodes_from(pred_labs, bpartite=1)
        edge_sets = []
        for row in range(r):
            if row in self.LG_ind_cc_pairing.keys():
                gt_lab = "gt_lab_" + str(self.LG_ind_cc_pairing[row])
                edges = np.argwhere(self.dsc[row, :]).flatten().tolist()
                pred_labs = [
                    "pred_lab_" + str(self.LP_ind_cc_pairing[ind]) for ind in edges
                ]
                gt_ind_2_preds = [(gt_lab, pred_lab) for pred_lab in pred_labs]
                edge_sets.extend(gt_ind_2_preds)
            else:
                pass  # This index LG has no overlapping pred
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
        colnames = ["label", "cent", "length", "volume", "label_cc", "fk"]
        dfs = []
        for i in range(len(self.df2)):
            cluster = self.df2.iloc[i]
            labels = cluster[label_key]
            fk = cluster["fk"]
            row = nbrhoods.loc[nbrhoods["label_cc"].isin(labels)]
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
        df_matched = pd.concat(dfs, axis=0)
        matched_labels = set(df_matched["label_cc"].tolist())
        unmatched_labels = set(nbrhoods["label_cc"].tolist()) - matched_labels
        df_unmatched = nbrhoods.loc[
            nbrhoods["label_cc"].isin(unmatched_labels), colnames
        ]
        df_output = pd.concat([df_matched, df_unmatched], axis=0)
        return df_output

    def insert_dsc_fks(self):
        self.LG.nbrhoods["dsc"] = float("nan")
        self.LG.nbrhoods["fk"] = -1
        self.LP.nbrhoods["fk"] = -2
        if self.empty_lm != "neither":
            pass
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

    def generate_fake_df2(self):
            fake_labels = {
                    "fk" :-3,    # # # any fk -3 rows should be removed
                "pred_label_cc": None,
                "gt_label_cc": None,
                "recompute_dsc": False,
                "dsc": None,
            }
            self.df2 = pd.DataFrame(fake_labels, index=[0])



    def process(self, debug=False):
            print("Processing {}".format(self.case_id))
            self.dust()
            self.compute_overlap_overall()
            if self.empty_lm == "neither":
                self.compute_overlap_perlesion()
                self.recompute_overlap_perlesion()
            self.insert_dsc_fks()
            # else:
            #     self.generate_fake_df2()
            self.gt_radiomics(debug)
            df = self.create_df_full()
            return df
        

class ScorerAdvancedITK(ScorerAdvanced):
    geometry_cls = LabelMapGeometryITK

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
        if not img_fn:
            assert do_radiomics == False, "To do_radiomics, provide img_fn"
        gt_fn, pred_fn = Path(gt_fn), Path(pred_fn)
        if case_id is None:
            self.case_id = info_from_filename(gt_fn.name, full_caseid=True)["case_id"]
        else:
            self.case_id = case_id
        self.gt = gt_fn
        self.pred = pred_fn
        self.img = load_labelmap_like_to_sitk(img_fn)[0] if img_fn else None

        self.LG = self.geometry_cls(gt_fn, ignore_labels_gt)
        self.LP = self.geometry_cls(pred_fn, ignore_labels_pred)
        if not results_folder:
            results_folder = "results"
        print("Dusting threshold {}".format(dusting_threshold))
        self.gt_fn = gt_fn
        self.pred_fn = pred_fn
        self.img_fn = img_fn
        self.params_fn = params_fn
        self.ignore_labels_gt = ignore_labels_gt
        self.ignore_labels_pred = ignore_labels_pred
        self.detection_threshold = detection_threshold
        self.dusting_threshold = dusting_threshold
        self.do_radiomics = do_radiomics
        self.save_matrices = save_matrices
        self.results_folder = results_folder

    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        prox_labels, prox_inds = self.get_neighbr_labels()
        self.LG_ind_cc_pairing = {a[0]: b[0] for a, b in zip(prox_inds, prox_labels)}
        self.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}

        self.dsc = np.zeros([max(1, len(self.LP)), max(1, len(self.LG))]).transpose()
        self.jac = np.copy(self.dsc)
        if self.empty_lm == "neither":
            args = [[self.LG.li_cc_sitk, self.LP.li_cc_sitk, *a] for a in prox_labels]
            d = multiprocess_multiarg(
                labels_overlap, args, 16, False, False, progress_bar=True
            )

            for i, sc in enumerate(d):
                ind_pair = list(prox_inds[i])
                self.dsc[ind_pair[0], ind_pair[1]] = sc[0]
                self.jac[ind_pair[0], ind_pair[1]] = sc[1]


    def insert_fks(self, nbrhoods, label_key):
        colnames = ["label", "cent", "major_axis", "volume_cc", "label_cc", "fk", "feret"]
        dfs = []
        for i in range(len(self.df2)):
            cluster = self.df2.iloc[i]
            labels = cluster[label_key]
            fk = cluster["fk"]
            row = nbrhoods.loc[nbrhoods["label_cc"].isin(labels)]
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
        df_matched = pd.concat(dfs, axis=0)
        matched_labels = set(df_matched["label_cc"].tolist())
        unmatched_labels = set(nbrhoods["label_cc"].tolist()) - matched_labels
        df_unmatched = nbrhoods.loc[
            nbrhoods["label_cc"].isin(unmatched_labels), colnames
        ]
        df_output = pd.concat([df_matched, df_unmatched], axis=0)
        return df_output


    def compute_overlap_overall(self):
        if self.empty_lm == "both":
            self.dsc_overall, self.jac_overall = np.nan, np.nan

        elif self.empty_lm == "one":
            self.dsc_overall, self.jac_overall = 0.0, 0.0
        else:
            self.dsc_overall, self.jac_overall = labels_overlap(
                self.LG.li_binary_sitk, self.LP.li_binary_sitk, 1, 1
            )
class ScorerAdvancedPT(ScorerAdvancedITK):
    geometry_cls = LabelMapGeometryPT


class BatchScorer:
    scorer_cls = ScorerAdvanced

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
        self.partial_df = partial_df
        self.output_fldr = output_fldr
        self.do_radiomics = do_radiomics
        self.debug = debug
        self.ignore_labels_gt = ignore_labels_gt
        self.ignore_labels_pred = ignore_labels_pred
        self.dusting_threshold = dusting_threshold
        self.preds_fldr = preds_fldr
        self.imgs_fldr = imgs_fldr
        gt_fns = self.filter_gt_fns(gt_fns, partial_df, exclude_fns)
        self.file_dicts = self.match_filenames(gt_fns)

    def process(self):
        self.dfs = []
        for fn_dict in self.file_dicts:
            gt_fn, pred_fn, img_fn = fn_dict.values()
            print("processing {}".format(gt_fn))
            try:
                self.S = self.scorer_cls(
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
                df = self.S.process(debug=self.debug)
            except Exception as e:
                        msg = "Error processing case_id={} gt_fn={} pred_fn={}".format(
                            self.S.case_id, self.S.gt_fn, self.S.pred_fn
                        )
                        logging.exception(msg)
                        raise RuntimeError(msg) from e

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
            img_fns = list(self.imgs_fldr.glob("*.*"))
        else:
            img_fns = None
        file_dicts = []
        for gt_fn in gt_fns:
            case_id = info_from_filename(gt_fn.name, full_caseid=True)["case_id"]
            pred_fn = [fn for fn in pred_fns if case_id in fn.name]
            if len(pred_fn) != 1:
                cprint(f"Multiple predictions for {pred_fn}",color="red")
                raise RuntimeError
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
        output_fn = self.output_fldr.name + "_thresh{0}mm.xlsx".format(
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
        # dfs_tmp.to_csv(self.output_fldr / "tmp{}.csv".format(self.output_suffix))
        dfs_tmp.to_csv(self.output_fn)

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


class BatchScorerITK(BatchScorer2):
    scorer_cls = ScorerAdvancedITK


class BatchScorerPT(BatchScorerITK):
    scorer_cls = ScorerAdvancedPT

if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc")
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-1018_fixed_mc")
    res_fldr = preds_fldr / "results"
    gt_fldr = Path("/s/xnat_shadow/crc/lms")
    fns = list(res_fldr.glob("*csv"))

    gt_fns = list(gt_fldr.glob("*"))

# %%

    lm_fn = "/home/ub/code/label_analysis/label_analysis/files/two_lesions.nrrd"
    cc_fn = "/home/ub/code/label_analysis/label_analysis/files/two_lesions_cc.mrk.json"
    point_fn = (
        "/home/ub/code/label_analysis/label_analysis/files/two_lesions_point.mrk.json"
    )
    lm = sitk.ReadImage(lm_fn)

    cc = load_json(cc_fn)
    point = load_json(point_fn)
    L = LabelMapGeometry(lm)
# %%
    L.nbrhoods
    L.fil.ComputeOrientedBoundingBoxOn()
    L.Execute(L.lm_cc)
    sitk.WriteImage(
        L.lm_cc, "/home/ub/code/label_analysis/label_analysis/files/two_lesions_cc.nii"
    )
# %%

    L = LabelMapGeometryITK(lm)
    L.lm_org.GetOrigin()
    L.lm_org.GetSpacing()
    L.fil.GetBoundingBox(1)

    print(org1, "\n", bb1)

    org2 = L.fil.GetOrientedBoundingBoxOrigin(2)
    bb2 = L.fil.GetOrientedBoundingBoxSize(2)
    bb2_end = [a + b for a, b in zip(org2, bb2)]
    # L.fil.GetOrientedBoundingBoxDirection(2)

    print(org2, "\n", bb2, "\n", bb2_end)

    print(org1)
    print(bb1)
    print(org2)
    print(bb2)
# %%

    mup = cc["markups"]
    len(mup)
    mup[0].keys()
# %%

# %%
    lm = to_int(L.li_cc)
    L.fil.GetOrientedBoundingBoxDirection(1)
    L.fil.GetBoundingBox(1)
    L.fil.Execute(lm)
    L.fil.GetOrientedBoundingBoxDirection(lm)
# %%
    pd.set_option("display.max_colwidth", None)
# %%
    dfs = []
    for fn in fns:
        df = pd.read_csv(fn)
        dfs.append(df)
    partial_df = None

    partial_df = pd.concat(dfs)
# %%
    # maybe_makedirs(preds_fldr/("results"))
    # "crc_CRC004_20190425_CAP1p5.nii.gz"; //"/s/fran_storage/predictions/lidc2/LITS-911/lung_020.nii.gz"; %%
    preds_nnunet_fldr = Path("/s/datasets_bkp/ark/")
    gt_fns = [fn for fn in gt_fns if is_sitk_file(fn)]

    # ub_df_fn = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/results/results_thresh1mm.xlsx"
    # results_df = pd.read_excel(ub_df_fn)
    # partial_df = results_df
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

# %%
# SECTION:-------------------- BATCH PT-------------------------------------------------------------------------------------- <CR>

    from utilz.helpers import info_from_filename
    from fran.managers.project import Project
    P = Project("kits")

    _, val = P.get_train_val_case_ids(0) 
    gt_fldr = Path("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms")
    gt_fns = sorted(gt_fldr.glob("*.pt"))
    val_fns =[fn for fn in gt_fns  if info_from_filename(fn.name, full_caseid=True)["case_id"] in val]
    len(val_fns)
    ignore_labels_gt = [1]
    ignore_labels_pred = [1]
    # preds_fldr = Path("/s/fran_storage/predictions/kits/KITS-n7")
    preds_fldr = Path("/s/fran_storage/predictions/kits/KITS-TW")
    preds_fldr = Path("/s/fran_storage/predictions/kits/KITS-bl")
    # gt_fns2 = gt_fns[:5]+ [Path("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits23_00030.pt")]
    pred_fns = [fn for fn in preds_fldr.glob("*") if info_from_filename(fn.name, full_caseid=True)["case_id"] in val]
    # pred_fn="/s/fran_storage/predictions/kits/KITS-n7/kits23_00011.pt"

    gt_fldr = Path("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms")
    gt_fns = list(gt_fldr.glob("*"))



    debug_  = True
    gt_fn =Path(" /r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits23_00002.pt")
    preds_fldr = Path("/s/fran_storage/predictions/kits/KITS-n7")

    find_matching_fn(gt_fn, preds_fldr)
# %%
    gt_fns = val_fns
    BP = BatchScorerPT(
        gt_fns=gt_fns,
        preds_fldr=preds_fldr,
        imgs_fldr=None,
        partial_df=None,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
        output_fldr=None,
        do_radiomics=False,
        dusting_threshold=1,
        debug=False,
        output_suffix=1
    )

    df = BP.process()
# %%
    S= ScorerAdvancedPT(gt_fn = gt_fn, pred_fn = pred_fn, case_id=None, ignore_labels_gt=[1], ignore_labels_pred=[1], do_radiomics=False, dusting_threshold=1)
    S.process()
    # gt_fns = [gt_fn]
# # %%
#
#
#         df_rad = pd.DataFrame(S.radiomics)
#         LG_out1 = S.LG.nbrhoods.rename(
#             mapper=lambda x: "gt_" + x if not x in ["dsc", "fk"] else x, axis=1
#         )
#         LG_out2 = df_rad.merge(
#             LG_out1, right_on="gt_label_cc", left_on="label", how="outer"
#         )
#
#         LP_out = S.LP.nbrhoods.rename(
#             mapper=lambda x: "pred_" + x if x != "fk" else x, axis=1
#         )
# # %%
#         if hasattr(S, "df2"):
#             dfs_all = [LG_out2, LP_out, S.df2[["fk", "dsc"]]]
#         else:
#             dfs_all = [LG_out2, LP_out]
#         df = reduce(lambda rt, lt: pd.merge(rt, lt, on="fk", how="outer"), dfs_all)
#
#         df["pred_fn"] = S.pred_fn
#         df["gt_fn"] = S.gt_fn
#         df["dsc_overall"], df["jac_overall"] = (
#             S.dsc_overall,
#             S.jac_overall,
#         )
#         df["gt_volume_total"] = S.LG.volume_total
#         df["pred_volume_total"] = S.LP.volume_total
#         df["case_id"] = df["case_id"].fillna(value=S.case_id)
#
    
# %%
    df=    BP.process()
# # %%
#
#             fn_dict = BP.file_dicts[0]
#             gt_fn, pred_fn, img_fn = fn_dict.values()
#             print("processing {}".format(gt_fn))
#             BP.S = BP.scorer_cls(
#                 gt_fn=gt_fn,
#                 img_fn=img_fn,
#                 pred_fn=pred_fn,
#                 ignore_labels_gt=BP.ignore_labels_gt,
#                 ignore_labels_pred=BP.ignore_labels_pred,
#                 case_id=None,
#                 save_matrices=False,
#                 do_radiomics=BP.do_radiomics,
#                 dusting_threshold=BP.dusting_threshold,
#             )
# # %%
#     S = BP.S
#
#     print("Processing {}".format(S.case_id))
#     S.dust()
#     S.compute_overlap_overall()
# # %%
#     if S.empty_lm == "neither":
#         S.compute_overlap_perlesion()
#         S.recompute_overlap_perlesion()
#         r =S.dsc.shape[0]
# # %%
#
#         print("Processing {}".format(S.case_id))
#         S.dust()
#         S.compute_overlap_overall()
#         if S.empty_lm == "neither":
#             S.compute_overlap_perlesion()
#             S.recompute_overlap_perlesion()
#         else:
#             S.generate_fake_df2()
#         S.gt_radiomics(False)
#
#         df = S.create_df_full()
# # %%
#         df.to_csv("tmp.csv")
# # %%
#         gt_labs = []
#         pred_labs = "pred_lab_" + S.LP.nbrhoods["label_cc"].astype(str)
#         G = nx.Graph()
#         G.add_nodes_from(gt_labs, bpartite=0)
#         G.add_nodes_from(pred_labs, bpartite=1)
#         edge_sets = []
# # %%
#         for row in range(r):
#             if row in S.LG_ind_cc_pairing.keys():
#                 gt_lab = "gt_lab_" + str(S.LG_ind_cc_pairing[row])
#                 edges = np.argwhere(S.dsc[row, :]).flatten().tolist()
#                 pred_labs = [
#                     "pred_lab_" + str(S.LP_ind_cc_pairing[ind]) for ind in edges
#                 ]
#                 gt_ind_2_preds = [(gt_lab, pred_lab) for pred_lab in pred_labs]
#                 edge_sets.extend(gt_ind_2_preds)
#             else:
#                 pass  # This index LG has no overlapping pred
#         G.add_edges_from(edge_sets)
#         con = nx.connected_components(G)
#         ccs = list(con)
#
# # %%
#         fk_gen = fk_generator(start=1)
#         gt_remaps = {}
#         pred_remaps = {}
#         fks = []
#         S.df2 = []
#         for cc in ccs:
#             fk = next(fk_gen)
#             fks.append(fk)
#             dici = {"fk": fk}
#             gt_labels = []
#             pred_labels = []
#             recompute_dsc = False
#             dsc = float("nan")
# # %%
#             if len(cc) > 2:
#                 recompute_dsc = True
#             elif len(cc) == 2:
#                 dsc = S.dsc_from_cc_pair(cc)
#             while cc:
#                 label = cc.pop()
#                 indx = re.findall(r"\d+", label)[0]
#                 indx = int(indx)
#                 if "gt" in label:
#                     gt_labels.append(indx)
#                     gt_remaps.update({indx: fk})
#                 else:
#                     pred_labels.append(indx)
#                     pred_remaps.update({indx: fk})
# # %%
#             labels = {
#                 "pred_label_cc": pred_labels,
#                 "gt_label_cc": gt_labels,
#                 "recompute_dsc": recompute_dsc,
#                 "dsc": dsc,
#             }
#             dici.update(labels)
# # %%
# # %%
#
#             S.df2.append(dici)
#         S.df2 = pd.DataFrame(S.df2)
#         S.LG._relabel(gt_remaps)
#         S.LP._relabel(pred_remaps)
#
#         redsc_labels = S.df2.loc[S.df2["recompute_dsc"] == True, "fk"].tolist()
#         prox_labels = [[a, a] for a in redsc_labels]
#
#         dsc_jac_multi = S._dsc_multilabel(prox_labels)
#
#         S.dsc_multi = [a[0] for a in dsc_jac_multi]
#         S.df2.loc[S.df2["fk"].isin(redsc_labels), "dsc"] = S.dsc_multi
#
#
# # %%
#     
#     # S.gt_radiomics(debug)
#     S.insert_dsc_fks()
#
#             S.gt_radiomics(debug_)
#             df  =S.create_df_full()
#             # df  =S.create_df_full()
# # # %%
#
#     S = BP.S
#     S.LP.nbrhoods
#     S.LP.labels
# # %%
#     lms = S.LP.unique_lms[0]
#     lms["lmap"].GetLabels()
# # def process(S, debug=False):
# #     print("Processing ")
#     S.dust()
#     S.compute_overlap_perlesion()
#     S.LP.is_empty()
#     S.LP.labels
# # %%
# #SECTION:-------------------- COMPUTE OL pER LESION--------------------------------------------------------------------------------------
#         print("Computing label jaccard and dice scores")
#         prox_labels, prox_inds = S.get_neighbr_labels()
#         S.LG_ind_cc_pairing = {a[0]: b[0] for a, b in zip(prox_inds, prox_labels)}
#         S.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}
#
#         S.dsc = np.zeros([max(1, len(S.LP)), max(1, len(S.LG))]).transpose()
#         S.jac = np.copy(S.dsc)
#         if S.empty_lm == "neither":
#             args = [[S.LG.li_cc_sitk, S.LP.li_cc_sitk, *a] for a in prox_labels]
#             d = multiprocess_multiarg(
#                 labels_overlap, args, 16, False, False, progress_bar=True
#             )
#
#             for i, sc in enumerate(d):
#                 ind_pair = list(prox_inds[i])
#                 S.dsc[ind_pair[0], ind_pair[1]] = sc[0]
#                 S.jac[ind_pair[0], ind_pair[1]] = sc[1]
#
# # %%
# df = S.LP.nbrhoods
#         bboxes_lg = S.LG.nbrhoods["bbox"]
#         bboxes_lp = S.LP.nbrhoods["bbox"]
#         prox_inds = proximity_indices2(bboxes_lg, bboxes_lp)
# # %%
#
#     S.make_one_to_one_dsc()
#     S.compute_overlap_overall()
#     S.cont_tables()
#     df = S.create_df_full()
# k# %%
# #SECTION:-------------------- TS--------------------------------------------------------------------------------------
#
#     fn_dict = BP.file_dicts[0]
#     gt_fn, pred_fn, img_fn = fn_dict.values()
#     print("processing {}".format(gt_fn))
#
# # %%
# #SECTION:-------------------- --------------------------------------------------------------------------------------            S.compute_overlap_overall()
