# %%
import sys
sys.path+=["/home/ub/code"]
from fastai.vision.augment import GetAttr, store_attr
import pandas as pd
import itertools as il
import functools as fl
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses
import six
from fran.utils.fileio import load_text, maybe_makedirs
from mask_analysis.helpers import *
from radiomics import featureextractor
from mask_analysis.labels import labels_overlap
from pathlib import Path
from fran.transforms.totensor import ToTensorT
from fran.utils.helpers import *
from fran.utils.imageviewers import *
import numpy as np

np.set_printoptions(linewidth=250)


def do_radiomics(img, mask, label, mask_fn, paramsFile=None):
    if not paramsFile:
        paramsFile = "configs/params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    featureVector = extractor.execute(img, mask, label=label)
    featureVector["label"] = featureVector["diagnostics_Configuration_Settings"][
        "label"
    ]
    featureVector["case_id"] = get_case_id_from_filename(None, mask_fn)
    featureVector["fn"] = mask_fn
    return featureVector


class LabelsGeometryAndRadiomics(GetAttr):
    """
    mask: binary sitk image
    img: greyscale sitk image. Default: None. If provided, processes per-label radiomics
    return n_labels and for each label: length, volume
    """

    def __init__(self, mask: sitk.Image, img=None, dusting_threshold=0.3):

        self.mask = to_cc(mask)
        self.img=img
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.fil.Execute(self.mask)

    def process(self):
        self.dust()
        output = self.lengths,self.volumes


    def getStats(self):
        keys = ['volume', 'diameters']
    def dust(self ):
        inds_small = [l < self.dusting_threshold for l in self.lengths]
        self.labels_small = list(il.compress(self.labels, inds_small))
        self._remove_labels(self.labels_small)

    def _remove_labels(self, labels):

        dici = {x: 0 for x in labels}
        self.mask = sitk.ChangeLabelLabelMap(to_label(self.mask), dici)
        self.mask = to_cc(self.mask)

    @property
    def lengths(self):
        self._lens = [
            max(self.fil.GetEquivalentEllipsoidDiameter(x)) for x in self.labels
        ]
        return self._lens

    @property
    def volumes(self):
        self._vols = [self.fil.GetPhysicalSize(x) * 1e-3 for x in self.labels]
        return self._vols

    @property
    def labels(self):
        return get_labels(self.mask)

    @property
    def n_labels(self):
        return len(self.labels)


class Scorer:
    '''
    input image, mask, and prediction to compute total dice, lesion-wise dice,  and lesion-wise radiomics (based on mask)
    '''
    
    def __init__(
        self,
        img_fn:Path,
        mask_fn,
        pred_fn,
        params_fn=None,
        remove_organ=True,
        detection_threshold=0.2,
        dusting_threshold=5,
        save=True,
        results_folder=None
        
    ) -> None:
        '''
        params_fn: specifies radiomics params
        '''
        
        self.case_id_fn = img_fn.name.split(".")[0]
        self.img, self.mask, self.pred = [
            sitk.ReadImage(fn) for fn in [img_fn, mask_fn, pred_fn]
        ]
        if remove_organ == True:
            self.mask, self.pred = map(remove_organ_mask, [self.mask, self.pred])
        self.mask_cc = to_cc(self.mask)
        self.pred_cc = to_cc(self.pred)
        if not results_folder: 
            results_folder ="results"
        store_attr()

    def process(self):
        print("Processing {}".format(self.case_id_fn))
        self.compute_stats()
        self.compute_overlap_perlesion()
        self.compute_overlap_overall()
        self.cont_tables()
        self.mask_radiomics()
        self.create_df()

    def compute_stats(self, dusting_threshold=None):
        if dusting_threshold is not None:
            self.dusting_threshold = dusting_threshold
        # predicted labels <threshold max_dia will be erased
        self.LM = LabelsGeometryAndRadiomics(self.mask_cc)
        self.LM.dust(self.dusting_threshold) # sanity check
        self.mask_cc_dusted = self.LM.mask
        self.LP = LabelsGeometryAndRadiomics(self.pred_cc)
        self.LP.dust(self.dusting_threshold)
        self.pred_cc_dusted = self.LP.mask

        self.labs_pred = self.LP.labels
        self.labs_mask = self.LM.labels

    def compute_overlap_overall(self):
        self.dsc_overall, self.jac_overall,_ = labels_overlap(self.pred,self.mask,1,1)
    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        # get jaccard and dice
        lab_inds = list(il.product(self.labs_mask, self.labs_pred))
        self.dsc = np.zeros((len(self.labs_pred), len(self.labs_mask))).transpose()
        self.jac = np.copy(self.dsc)
        args = [[self.mask_cc_dusted, self.pred_cc_dusted, *a] for a in lab_inds]
        d = multiprocess_multiarg(
            labels_overlap, args, 16, False, False, progress_bar=True
        )  # multiprocess i s slow

        for sc in d:
            ind_pair = sc[2]
            self.dsc[ind_pair] = sc[0]
            self.jac[ind_pair] = sc[1]

    def cont_tables(self ):
        """
        param detection_threshold: jaccard b/w lab_mask and lab_ored below this will be counted as a false negative
        """
        n_predicted = np.sum(self.dsc> self.detection_threshold, 1)
        self.detected = n_predicted > 0
        tt = self.dsc<= self.detection_threshold
        self.fp_pred_labels = list(np.where(np.all(tt == True, 0))[0] + 1)

    def mask_radiomics(self):
        self.radiomics = radiomics_multiprocess(self.img,self.mask_cc_dusted,self.labs_mask,self.params_fn)

    def create_df(self):
        df = pd.DataFrame(self.radiomics)
        df["detected"] = self.detected
        names_prefix = lambda jd, seq: [
            "_".join([jd, "pred", "label", str(i)]) for i in seq
        ]
        jac_labels = names_prefix("jac", self.labs_pred)
        dsc_labels = names_prefix("dsc", self.labs_pred)
        jac_df = pd.DataFrame(data=self.jac, columns=jac_labels)
        dsc_df = pd.DataFrame(data=self.dsc, columns=dsc_labels)
        self.df_final = pd.concat([df, dsc_df, jac_df], axis=1)
        self.df_final["fp_pred_labels"] = [
            self.fp_pred_labels,
        ] * len(self.labs_mask)
        self.df_final['dsc_overall'], self.df_final['jac_overall'] = self.dsc_overall,self.jac_overall
        if self.save == True:
            self.df_final.to_csv(self.results_folder/("res_{}_.csv".format(self.case_id_fn)), index=False)
            sitk.WriteImage(self.pred_cc_dusted,self.results_folder/(self.case_id_fn+"_pred.nrrd"))
            sitk.WriteImage(self.mask_cc_dusted,self.results_folder/(self.case_id_fn+"_mask.nrrd"))

    @property
    def results_folder(self):
        """The results_folder property."""
        return self._results_folder
    @results_folder.setter
    def results_folder(self, value):
        self._results_folder = Path(value)
        maybe_makedirs(self._results_folder)


def radiomics_multiprocess(img, mask,labels,params_fn):
        print("Computing mask label radiomics")
        args = [
            [img, mask, label, mask_fn,params_fn] for label in labels
        ]
        radiomics = multiprocess_multiarg(
            do_radiomics, args, num_processes=np.maximum(len(args),1), multiprocess=True
        )
        return radiomics



fname_no_ext = lambda fn : fn.split('.')[0]
# %%
if __name__ == "__main__":
    preds_fldr = Path(
        # "/s/fran_storage/predictions/lits/ensemble_LITS-265_LITS-255_LITS-270_LITS-271_LITS-272/"
    "/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
    )

# %%
    masks_fldr = Path("/s/datasets_bkp/lits_segs_improved/masks/")
    imgs_fldr = Path("/s/datasets_bkp/lits_segs_improved/images/")
    masks = list(masks_fldr.glob("*"))
# %%
    dfs=[]
# %%
    for n in range(len(masks)):
        mask_fn = masks[n]
        print("processing {}".format(mask_fn))
        pred_fn = [fn for fn in preds_fldr.glob("*") if fname_no_ext(mask_fn.name)== fname_no_ext(fn.name)][0]
        img_fn = [fn for fn in imgs_fldr.glob("*") if fname_no_ext(mask_fn.name)== fname_no_ext(fn.name)][0]
        S = Scorer(img_fn, mask_fn, pred_fn)
        S.process()
        dfs.append(S.df_final)
# %%
    df_final = pd.concat(dfs)
    df_final.to_csv("results/results.csv",index=False)
# %%
    view_sitk(S.pred_cc_dusted,S.mask_cc_dusted,data_types=['mask','mask'])
    sitk.WriteImage(S.pred_cc_dusted,Path(S.results_folder)/("pred_cc.nrrd"))
    sitk.WriteImage(S.mask_cc_dusted,Path(S.results_folder)/("mask_cc.nrrd"))
# %%
    pred = to_int(S.pred)
    mask = to_int(S.mask)
    fil = sitk.LabelOverlapMeasuresImageFilter()
    fil.Execute(pred,mask)

# %%
    get_labels(mask)
# %%
    S.dsc_overall, S.jac_overall,_ = labels_overlap(S.pred,S.mask,1,1)
# %%

    detection_threshold=0.25
    n_predicted = np.sum(S.jac > detection_threshold, 1)
    S.detected = n_predicted > 0
    tt = S.dsc<= detection_threshold
    S.fp_pred_labels = list(np.where(np.all(tt == True, 0))[0] + 1)
# %%
