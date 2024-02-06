# %%
import shutil
from functools import reduce
import sys
from os import remove
from types import NotImplementedType

from tqdm.notebook import tqdm

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from fastcore.basics import GetAttr, store_attr
from label_analysis.helpers import *
from radiomics import featureextractor, getFeatureClasses

from fran.transforms.totensor import ToTensorT
from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.string import (find_file, info_from_filename, match_filenames,
                               strip_extension, strip_slicer_strings)

np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def keep_largest(onedarray):
    largest = onedarray.max()
    onedarray[onedarray < largest] = 0
    return onedarray


@astype([5, 5], [0, 1])
def labels_overlap(gt_cc, pred_cc, lab_gt, lab_pred):
    gt_all_labels = get_labels(gt_cc)
    assert (
        lab_gt in gt_all_labels
    ), "Label {} is not present in the Groundtruth ".format(lab_gt)
    mask2 = single_label(gt_cc, lab_gt)
    pred2 = single_label(pred_cc, lab_pred)
    fil = sitk.LabelOverlapMeasuresImageFilter()
    a, b = map(to_int, [mask2, pred2])
    fil.Execute(a, b)
    dsc, jac = fil.GetDiceCoefficient(), fil.GetJaccardCoefficient()
    return dsc, jac


def remap_single_label(lm, target_label, starting_ind):
    lm_tmp = single_label(lm, target_label)
    lm_tmp = to_cc(lm_tmp)
    labs = get_labels(lm_tmp)
    remapping = {l: l + starting_ind for l in labs}
    lm_tmp = relabel(lm_tmp, remapping)
    lm_tmp = to_label(lm_tmp)
    return lm_tmp, list(remapping.values())


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
        df = get_1lbl_nbrhoods(labelmap, label, dusting_threshold=dusting_threshold)
        dfs.append(df)
    df_final = pd.concat(dfs)
    df_final.reset_index(inplace=True, drop=True)
    return df_final


def do_radiomics(img, mask, label, mask_fn, paramsFile=None):
    if not paramsFile:
        paramsFile = "mask_analysis/configs/params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

    featureVector = {}
    featureVector["case_id"] = info_from_filename(mask_fn.name)["case_id"]
    featureVector["fn"] = mask_fn
    featureVector2 = extractor.execute(img, mask, label=label)
    featureVector["label"] = featureVector2["diagnostics_Configuration_Settings"][
        "label"
    ]
    featureVector.update(featureVector2)
    return featureVector


class LabelMapGeometry(GetAttr):
    """
    lm: can have multiple labels
    img: greyscale sitk image. Default: None. If provided, processes per-label radiomics
    """

    _default = "fil"

    def __init__(self, lm: sitk.Image, ignore_labels=[1], img=None):
        self.fil = sitk.LabelShapeStatisticsImageFilter()

        if len(ignore_labels) > 0:
            remove_labels = {l: 0 for l in ignore_labels}
            lm = relabel(lm, remove_labels)
        self.lm_org = lm
        self.create_lm_binary()
        self.create_lm_cc()  # creates ordered labelmap from original labels and a key mapping
        self.execute_filter()
        self.calc_geom()

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
        try:
            self.nbrhoods = []
            for key, value in self.key.items():
                centroid = self.GetCentroid(key)
                radius = self.GetEquivalentSphericalRadius(key)
                dici = {
                    "label": value,
                    "label_cc": key,
                    "cent": centroid,
                    "rad": radius,
                    "length": self.lengths[key],
                    "volume": self.volumes[key],
                }
                self.nbrhoods.append(dici)
            self.nbrhoods = pd.DataFrame(self.nbrhoods)
        except:
            self.nbrhoods = pd.DataFrame(
                columns=["label", "label_cc", "cent", "rad", "length", "volume"]
            )
        return self.nbrhoods

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
        self.lm_cc = to_label(self.lm_cc)
        self.lm_cc = sitk.ChangeLabelLabelMap(self.lm_cc, remapping)
        self.execute_filter()

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
        self._lengths = {
            x: max(self.GetEquivalentEllipsoidDiameter(x)) for x in self.labels
        }
        return self._lengths

    @property
    def ferets(self):
        self.fil.ComputeFeretDiameterOn()
        self.execute_filter()
        self._ferets = {x: self.GetFeretDiameter(x) for x in self.labels}
        return self._ferets

    @property
    def centroids(self):
        return [np.array(self.fil.GetCentroid(x)) for x in self.labels]


class Scorer:
    """
    input image, mask, and prediction to compute total dice, lesion-wise dice,  and lesion-wise radiomics (based on mask)
    """

    def __init__(
        self,
        gt_fn: Union[str,Path],
        pred_fn: Union[str,Path],
        img_fn: Union[str,Path]=None,
        params_fn=None,
        ignore_labels_gt=[1],
        ignore_labels_pred=[1],
        detection_threshold=0.2,
        dusting_threshold=3,
        do_radiomics=False,
        save_matrices=True,
        results_folder=None,
    ) -> None:
        """
        params_fn: specifies radiomics params
        save_matrices: if true, save full dsc and jac matrices in a separate file
        """
        if not img_fn: assert do_radiomics==False, "To do_radiomics, provide img_fn"
        gt_fn, pred_fn =  Path(gt_fn), Path(pred_fn)
        self.case_id = gt_fn.name.split(".")[0]
        self.gt, self.pred = [
            sitk.ReadImage(fn) for fn in [gt_fn, pred_fn]
        ]
        self.img = sitk.ReadImage(img_fn) if img_fn else None

        self.LG = LabelMapGeometry(self.gt, ignore_labels_gt)
        self.LP = LabelMapGeometry(self.pred, ignore_labels_pred)
        if not results_folder:
            results_folder = "results"
        store_attr()

    def process(self, debug=False):
        print("Processing {}".format(self.case_id))
        self.compute_stats()
        self.gt_radiomics(debug)
        self.compute_overlap_perlesion()
        self.make_one_to_one_dsc()
        self.compute_overlap_overall()
        self.cont_tables()
        if len(self.labs_gt) > 0:
            return self.create_df_full()
        else:
            return self.create_dummy_df()

    def compute_stats(self):
        # predicted labels <threshold max_dia will be erased
        self.LG.dust(self.dusting_threshold)  # sanity check
        self.gt_cc_dusted = self.LG.lm_cc
        self.LP.dust(self.dusting_threshold)
        self.pred_cc_dusted = self.LP.lm_cc

        self.labs_pred = self.LP.labels
        self.labs_gt = self.LG.labels

    def compute_overlap_overall(self):
        if len(self.LG.lengths) == 0 and len(self.LP.lengths) == 0:
            self.dsc_overall, self.jac_overall = 1.0, 1.0

        elif (len(self.LG) == 0) ^ (len(self.LP) == 0):
            self.dsc_overall, self.jac_overall = 0.0, 0.0
        else:
            self.dsc_overall, self.jac_overall = labels_overlap(
                self.LG.lm_binary, self.LP.lm_binary, 1, 1
            )

    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        # get jaccard and dice

        prox_labels, prox_inds = self.get_neighbr_labels()
        self.dsc = np.zeros(
            [max(1, len(self.LP)), max(1, len(self.LG))]
        ).transpose()  # max(1,x) so that an empty matrix is not created
        self.jac = np.copy(self.dsc)
        args = [[self.gt_cc_dusted, self.pred_cc_dusted, *a] for a in prox_labels]

        d = multiprocess_multiarg(
            labels_overlap, args, 16, False, False, progress_bar=True
        )  # multiprocess i s slow

        # this pairing is a limited number of indices (and corresponding labels) which are in neighbourhoods between LG and LP
        self.LG_ind_cc_pairing = {a[0]: b[0] for a, b in zip(prox_inds, prox_labels)}
        self.LP_ind_cc_pairing = {a[1]: b[1] for a, b in zip(prox_inds, prox_labels)}
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
        if len(self.labs_gt) == 0 or self.do_radiomics == False:
            print("No radiomicss being done. ")
            dic = {}
            dic["case_id"] = self.case_id
            dic["gt_fn"] = self.gt_fn
            self.radiomics = [dic] * max(1, len(self.labs_gt))

        else:
            self.radiomics = radiomics_multiprocess(
                self.img,
                self.gt_cc_dusted,
                self.labs_gt,
                self.gt_fn,
                self.params_fn,
                debug,
            )

    def create_dummy_df(self):

        colnames = [
            "case_id",
            "gt_fn",
            "pred_fn",
            "gt_label",
            "gt_label_cc",
            "gt_length",
            "gt_volume",
            "gt_volume_total",
            "pred_label",
            "pred_label_cc",
            "pred_length",
            "pred_volume",
            "pred_volume_total",
            "dsc",
            "fp_pred_labels",
            "dsc_overall",
            "jac_overall",
        ]

        values = [
            [
                self.case_id,
                self.gt_fn,
                self.pred_fn,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                self.fp_pred_labels,
                0,
                0,
            ]
        ]
        # dici = {'case_id' : self.case_id, 'gt_fn':self.gt_fn,'pred_fn':self.pred_fn, 'gt_label':0, 'gt_label_cc':0, 'gt_length':0, 'gt_volume':0, 'pred_label':0 ,
        #         'pred_label_cc' : 0, 'pred_length': 0, 'pred_volume':0,  'dsc':0,'fp_pred_labels': str(self.fp_pred_labels), 'dsc_overall':0, 'jac_overall':0}
        df = pd.DataFrame(values, columns=colnames)
        # colnames = ['case_id' , 'gt_fn','pred_fn', 'gt_label', 'gt_label_cc', 'gt_length', 'gt_volume', 'pred_label' , 'pred_label_cc', 'pred_length', 'pred_volume',  'dsc',
        #    'fp_pred_labels', 'dsc_overall', 'jac_overall']
        # vals = [ self.case_id , self.gt_fn, self.pred_fn, 0,0,0,0,0,0,0]  +[self.fp_pred_labels,0,0]
        # df = pd.DataFrame({col:[val] for col,val in zip (colnames,vals)})
        return df

    def create_df_full(self):

        # prox_labels, prox_inds = self.get_neighbr_labels()
        pos_inds = np.argwhere(self.dsc_single_vals)
        pos_inds_gt = pos_inds[:, 0]
        pos_inds_pred = pos_inds[:, 1]

        gt_all = set(self.LG.nbrhoods.index)
        gt_left_over = gt_all.difference(set(pos_inds_gt))
        gt_inds = list(pos_inds_gt) + list(gt_left_over)
        lg_short = self.LG.nbrhoods[["label", "label_cc", "length", "volume"]].iloc[
            gt_inds
        ]
        lg_short.reset_index(drop=True, inplace=True)
        lg_short.rename(mapper=lambda x: "gt_" + x, axis=1, inplace=True)
        lp_short = self.LP.nbrhoods[["label", "label_cc", "length", "volume"]].iloc[
            pos_inds_pred
        ]
        lp_short.reset_index(drop=True, inplace=True)
        lp_short = append_empty_rows(lp_short, len(gt_left_over))
        lp_short.rename(mapper=lambda x: "pred_" + x, axis=1, inplace=True)

        df_rad = pd.DataFrame(self.radiomics)
        df = pd.concat([df_rad, lg_short, lp_short], axis=1)

        try:
            dscs = np.apply_along_axis(self.get_vals_from_indpair, 1, pos_inds)
            dscs = np.append(dscs, [0] * len(gt_left_over))
            df["dsc"] = dscs
        except:
            df["dsc"] = 0.0
        df["pred_fn"] = self.pred_fn
        df["fp_pred_labels"] = [
            self.fp_pred_labels,
        ] * np.maximum(1, len(self.LG))

        df["dsc_overall"], df["jac_overall"] = (
            self.dsc_overall,
            self.jac_overall,
        )
        df['gt_volume_total'] = self.LG.volume_total
        df['pred_volume_total'] = self.LP.volume_total
        if self.save_matrices == True:
            self.save_overlap_matrices()
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


def radiomics_multiprocess(img, mask, labels, mask_fn, params_fn, debug=False):
    print("Computing mask label radiomics")
    args = [[img, mask, label, mask_fn, params_fn] for label in labels]
    radiomics = multiprocess_multiarg(
        do_radiomics,
        args,
        num_processes=np.maximum(len(args), 1),
        multiprocess=True,
        debug=debug,
    )
    return radiomics


class BatchScorer():
    def __init__(self,  gt_fns: Union[Path,list],
    preds_fldr: Path,
    imgs_fldr: Path = None,
    partial_df:pd.DataFrame=None,
    exclude_fns=[],
    output_fldr=None,
    do_radiomics=False,
    dusting_threshold=3,
    debug=False,
    ignore_labels_gt=[],
    ignore_labels_pred=[1]
):

        if isinstance(gt_fns,Path):
            self.gt_fns = list(gt_fns.glob("*"))
        self.exclude_fns = self.create_exclusion_list(partial_df,exclude_fns)
        if len(exclude_fns) > 0:
            self.gt_fns = [fn for fn in self.gt_fns if fn not in exclude_fns]
        store_attr('partial_df,output_fldr,do_radiomics,debug,ignore_labels_gt, ignore_labels_pred,dusting_threshold')

    def process(self):
        dfs = []
        for fn_dict in self.file_dicts:
            gt_fn, pred_fn, img_fn = fn_dict.values()
            print("processing {}".format(gt_fn))
            S = Scorer(
                gt_fn = gt_fn,
                img_fn = img_fn,
                pred_fn=pred_fn,
                ignore_labels_gt=self.ignore_labels_gt,
                ignore_labels_pred=self.ignore_labels_pred,
                save_matrices=False,
                do_radiomics=self.do_radiomics,
                dusting_threshold=self.dusting_threshold,
            )
            df = S.process(debug=self.debug)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        print("Saving results to {}".format(self.output_fn))
        df.to_csv(self.output_fn, index=False)
        return df

    def create_file_tuples(self,gt_fns,preds_fldr,imgs_fldr):
        pred_fns = list(preds_fldr.glob("*"))
        if imgs_fldr: img_fns = list(imgs_fldr.glob("*")) 
        else: img_fns=  None
        self.file_dicts = []
        for gt_fn in gt_fns:
                pred_fn = [
                    fn for fn in pred_fns if match_filenames(gt_fn.name, fn.name) == True
                ]
                if len(pred_fn) != 1:
                    tr()
                else:
                    pred_fn = pred_fn[0]
                fn_dict = {'gt_fn':gt_fn, 'pred_fn':pred_fn, 'img_fn':None}
                if img_fns:
                    img_fns = [
                        fn for fn in imgs if match_filenames(gt_fn.name, fn.name) == True
                    ]
                    if len(img_fns) != 1:
                        tr()
                    else:
                        img_fn = img_fns[0]
                    fn_dict['img_fn'] = img_fn
                self.file_dicts.append(fn_dict)



    def create_exclusion_list(self,df,exclude_fns):
        raise NotImplemented

    @property
    def output_fn(self):
        output_fldr = self.preds_fldr/("results")
        maybe_makedirs(output_fldr)
        output_fn = output_fldr.name+"_thresh{}mm_results.csv".format(self.dusting_threshold)
        output_fn = output_fldr / output_fn
        return output_fn


# %%
if __name__ == "__main__":
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/")

# %%
    gt_fldr = Path("/s/xnat_shadow/crc/completed/masks")
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    gt_fns = list(gt_fldr.glob("*"))
    gt_fn = find_file(gt_fns, "CRC003")
    pred_fn= find_file(preds_fldr.glob("*"), "CRC003")

# %%
    S = Scorer(gt_fn,pred_fn,ignore_labels_gt=[],ignore_labels_pred=[1],save_matrices=False)
    df = S.process()
# %%
    df = process_gt_pred_folders(gt_fns, preds_fldr, imgs_fldr, debug=True, output_fn="results/df_6.2.csv")
# %%

    exclude_fns = []
    preds = list(preds_fldr.glob("*"))
    imgs = list(imgs_fldr.glob("*"))
    dfs = []
    ignore_labels_gt = []
    for gt_fn in gt_fns:
        if gt_fn not in exclude_fns:
            print("processing {}".format(gt_fn))
            pred_fns = [
                fn for fn in preds if match_filenames(gt_fn.name, fn.name) == True
            ]
            if len(pred_fns) != 1:
                tr()
            else:
                pred_fn = pred_fns[0]

            img_fns = [
                fn for fn in imgs if match_filenames(gt_fn.name, fn.name) == True
            ]
            if len(img_fns) != 1:
                tr()
            else:
                img_fn = img_fns[0]

            S = Scorer(
                img_fn,
                gt_fn,
                pred_fn,
                ignore_labels_gt=ignore_labels_gt,
                save_matrices=False,
                do_radiomics=do_radiomics,
            )

            df = S.process(debug=debug)
            dfs.append(df)

# %%
        if not img_fn: assert do_radiomics==False, "To do_radiomics, provide img_fn"
        gt_fn, pred_fn =  Path(gt_fn), Path(pred_fn)
        S.case_id = gt_fn.name.split(".")[0]
        gt, pred = [
            sitk.ReadImage(fn) for fn in [gt_fn, pred_fn]
        ]
        img = sitk.ReadImage(img_fn) if img_fn else None

        LG = LabelMapGeometry(gt, ignore_labels_gt)
        LP = LabelMapGeometry(pred, ignore_labels_pred)

# %%
        pos_inds = np.argwhere(S.dsc_single_vals)
        fk = list(np.arange(pos_inds.shape[0]))
        pos_inds_gt = pos_inds[:, 0]
        pos_inds_pred = pos_inds[:, 1]

        gt_all = set(S.LG.nbrhoods.index)
        fn_inds = gt_all.difference(set(pos_inds_gt))
        gt_inds = list(pos_inds_gt) + list(fn_inds)

        pred_all = set(S.LP.nbrhoods.index)
        fp_inds = pred_all.difference(set(pos_inds_pred))
# %%
        S.LG.nbrhoods['fk']=np.nan
        S.LG.nbrhoods.loc[pos_inds_gt,'fk'] = fk

        S.LP.nbrhoods['fk'] = np.nan
        S.LP.nbrhoods.loc[pos_inds_pred,'fk'] =fk
        S.LP.nbrhoods.rename(mapper=lambda x:  "pred_" + x  if x!="fk" else x, axis=1, inplace=True)

        S.LG.nbrhoods.iloc[pos_inds_gt]['pred_inds']= pos_inds_pred
        S.LG.nbrhoods.rename(mapper=lambda x:  "gt_" + x  if x!="fk" else x, axis=1, inplace=True)

        dfs_all = df_rad, S.LG.nbrhoods, S.LP.nbrhoods,dscs
        df = pd.merge(df_rad, S.LG.nbrhoods, S.LP.nbrhoods,dscs, how="outer", on="fk")
        df = reduce(lambda rt, lt: pd.merge(rt, lt, on="fk", how="outer"), dfs_all)

        pd.merge(df_rad,a, how="outer",on="fk").shape

        lp_short.reset_index(drop=True, inplace=True)
        lp_short = append_empty_rows(lp_short, len(fn_inds))
        lp_short.rename(mapper=lambda x: "pred_" + x, axis=1, inplace=True)

        df_rad = pd.DataFrame(S.radiomics)
        df_rad['fk'] = fk

        dscs = pd.DataFrame([[0,ff] for ff in fk],columns=['dsc', 'fk'])
        try:
            dscs ['dsc']= np.apply_along_axis(S.get_vals_from_indpair, 1, pos_inds)
        except:
            pass
        df["pred_fn"] = S.pred_fn
        df["fp_pred_labels"] = [
            S.fp_pred_labels,
        ] * np.maximum(1, len(S.LG))

        df["dsc_overall"], df["jac_overall"] = (
            S.dsc_overall,
            S.jac_overall,
        )
        df['gt_volume_total'] = S.LG.volume_total
        df['pred_volume_total'] = S.LP.volume_total
        if S.save_matrices == True:
            S.save_overlap_matrices()
# %%


# %%
