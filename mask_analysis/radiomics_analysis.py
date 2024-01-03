# %%
import sys
sys.path+=["/home/ub/code"]
from fastcore.basics import GetAttr, store_attr
import pandas as pd
import itertools as il
import functools as fl
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses
import six
from fran.utils.fileio import  maybe_makedirs
from mask_analysis.helpers import *
from radiomics import featureextractor
from mask_analysis.labels import labels_overlap
from pathlib import Path
from fran.transforms.totensor import ToTensorT
from fran.utils.helpers import *
from fran.utils.string import info_from_filename, match_filenames, strip_slicer_strings
from fran.utils.imageviewers import *
import numpy as np

np.set_printoptions(linewidth=250)


def do_radiomics(img, mask, label, mask_fn, paramsFile=None):
    if not paramsFile:
        paramsFile = "mask_analysis/configs/params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

    featureVector={}
    featureVector["case_id"] = info_from_filename(mask_fn.name)['case_id']
    featureVector["fn"] = mask_fn
    featureVector2 = extractor.execute(img, mask, label=label)
    featureVector["label"] = featureVector2["diagnostics_Configuration_Settings"][
        "label"
    ]
    featureVector.update(featureVector2)
    return featureVector


class LabelsGeometryAndRadiomics(GetAttr):
    """
    mask: binary sitk image
    img: greyscale sitk image. Default: None. If provided, processes per-label radiomics
    return n_labels and for each label: length, volume
    """
    _default = 'fil'

    def __init__(self, mask: sitk.Image, img=None ):

        store_attr('img')
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.mask = to_cc(mask)
        self.fil.Execute(self.mask)


    def dust(self,dusting_threshold ):
        inds_small = [l < dusting_threshold for l in self.lengths.values()]
        self.labels_small = list(il.compress(self.labels, inds_small))
        self._remove_labels(self.labels_small)

    def _remove_labels(self, labels):

        dici = {x: 0 for x in labels}
        self.mask = sitk.ChangeLabelLabelMap(to_label(self.mask), dici)
        self.mask = to_cc(self.mask)
        self.fil.Execute(self.mask)

    def relabel(self,remapping):

        self.mask= to_label(self.mask)
        self.mask = sitk.ChangeLabelLabelMap(self.mask,remapping)
        self.mask= to_int(self.mask)
        self.fil.Execute(self.mask) # update filter readings


    @property
    def labels(self): return self.GetLabels()

    @property
    def n_labels(self):
        return len(self.labels)


    @property
    def volumes(self): 
          self._volumes= {x:self.GetPhysicalSize(x) * 1e-3 for x in self.labels}
          return self._volumes

    @property
    def volume_total(self):
            return sum(self.volumes.values())

    @property
    def lengths(self): 
          self._lengths = {x:max(self.GetEquivalentEllipsoidDiameter(x))  for x in self.labels}
          return self._lengths

    @property 
    def centroids(self):return  [self.fil.GetCentroid(x) for x in self.labels]
        


class Scorer:
    '''
    input image, mask, and prediction to compute total dice, lesion-wise dice,  and lesion-wise radiomics (based on mask)
    '''
    
    def __init__(
        self,
        img_fn:Path,
        gt_fn,
        pred_fn,
        params_fn=None,
        labels=[1,2],
        target_label=2, # typically, organ is label 1, tumour  is label 2 in liver or kidney projects.
        detection_threshold=0.2,
        dusting_threshold=5,
        save=True,
        results_folder=None
        
    ) -> None:
        '''
        params_fn: specifies radiomics params
        '''
        
        self.case_id_fn = img_fn.name.split(".")[0]
        self.img, self.gt, self.pred = [
            sitk.ReadImage(fn) for fn in [img_fn, gt_fn, pred_fn]
        ]
        self.gt, self.pred = remove_organ_label(self.gt,False),remove_organ_label(self.pred,False)
        self.gt_cc = to_cc(self.gt)
        self.pred_cc = to_cc(self.pred)
        if not results_folder: 
            results_folder ="results"
        store_attr()

    def process(self,debug=False):
        print("Processing {}".format(self.case_id_fn))
        self.compute_stats()
        self.compute_overlap_perlesion()
        self.compute_overlap_overall()
        self.cont_tables()
        self.gt_radiomics(debug)
        self.create_df()

    def compute_stats(self):
        # predicted labels <threshold max_dia will be erased
        self.LG = LabelsGeometryAndRadiomics(self.gt_cc)
        self.LG.dust(self.dusting_threshold) # sanity check
        self.gt_cc_dusted = self.LG.mask
        self.LP = LabelsGeometryAndRadiomics(self.pred_cc)
        self.LP.dust(self.dusting_threshold)
        self.pred_cc_dusted = self.LP.mask

        self.labs_pred = self.LP.labels
        self.labs_gt = self.LG.labels

    def compute_overlap_overall(self):
        if len(self.LG.lengths)==0 or len(self.LP.lengths)==0:
            self.dsc_overall, self.jac_overall=0.0,0.0
        else:
            self.dsc_overall, self.jac_overall,_ = labels_overlap(self.pred,self.gt,1,1)
    def compute_overlap_perlesion(self):
        print("Computing label jaccard and dice scores")
        # get jaccard and dice
        lab_inds = list(il.product(self.labs_gt, self.labs_pred))
        self.dsc = np.zeros((len(self.labs_pred), len(self.labs_gt))).transpose()
        self.jac = np.copy(self.dsc)
        args = [[self.gt_cc_dusted, self.pred_cc_dusted, *a] for a in lab_inds]
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
        if n_predicted.size>0:
            self.detected = n_predicted > 0
        else:
            self.detected=None
        tt = self.dsc<= self.detection_threshold
        self.fp_pred_labels = list(np.where(np.all(tt == True, 0))[0] + 1)

    def gt_radiomics(self,debug):
        if len(self.labs_gt)==0:
            print("No gt labels. No radiomicss being done. ")
            dic={}
            dic["case_id"] = info_from_filename(gt_fn.name)['case_id']
            dic["fn"] = gt_fn
            self.radiomics = [dic]

        else:
            self.radiomics= radiomics_multiprocess(self.img,self.gt_cc_dusted,self.labs_gt,self.gt_fn,self.params_fn,debug)

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
        ] * np.maximum(1,len(self.labs_gt))

        self.df_final['dsc_overall'], self.df_final['jac_overall'] = self.dsc_overall,self.jac_overall
        if self.save == True:
            self.df_final.to_csv(self.results_folder/("res_{}_.csv".format(self.case_id_fn)), index=False)
            sitk.WriteImage(self.pred_cc_dusted,self.results_folder/(self.case_id_fn+"_pred.nrrd"))
            sitk.WriteImage(self.gt_cc_dusted,self.results_folder/(self.case_id_fn+"_gt.nrrd"))

    @property
    def results_folder(self):
        """The results_folder property."""
        return self._results_folder
    @results_folder.setter
    def results_folder(self, value):
        self._results_folder = Path(value)
        maybe_makedirs(self._results_folder)


def radiomics_multiprocess(img, mask,labels,mask_fn,params_fn,debug=False):
        print("Computing mask label radiomics")
        args = [
            [img, mask, label, mask_fn,params_fn] for label in labels
        ]
        radiomics = multiprocess_multiarg(
            do_radiomics, args, num_processes=np.maximum(len(args),1), multiprocess=True, debug=debug
        )
        return radiomics



fname_no_ext = lambda fn : fn.split('.')[0]

# %%
if __name__ == "__main__":
    preds_fldr = Path(
    # "/s/fran_storage/predictions/lits/ensemble_LITS-499_LITS-500_LITS-501_LITS-502_LITS-503/"
        # "/s/fran_storage/predictions/lits32/LIT-143_LIT-150_LIT-149_LIT-153_LIT-161"
        "/s/fran_storage/predictions/lits32/LITS-630_LITS-633_LITS-632_LITS-647_LITS-650"
    )

# %%
    gt_fldr = Path("/s/xnat_shadow/crc/test/labels_withliver")
    imgs_fldr = Path("/s/xnat_shadow/crc/test/images/finalised")
    
    gts = list(gt_fldr.glob("*"))
    preds = list(preds_fldr.glob("*"))
    imgs=list(imgs_fldr.glob("*"))
    gt_fn = Path("/s/xnat_shadow/crc/test/labels_withliver/crc_CRC010_20171117_Abdomen3p0I30f3.nii.gz")
    img_fn = Path("/s/xnat_shadow/crc/test/images/finalised/crc_CRC010_20171117_Abdomen3p0I30f3.nii.gz")
    pred_fn = Path("/s/fran_storage/predictions/lits32/LIT-143_LIT-150_LIT-149_LIT-153_LIT-161/crc_CRC010_20171117_Abdomen3p0I30f3.nii.gz")
# %%
    dfs=[]
# %%
    debug=False
    for n in range(0,len(gts)):
        gt_fn = gts[n]

        print("processing {}".format(gt_fn))
        pred_fns = [fn for fn in preds if match_filenames(gt_fn.name,fn.name)==True]
        if len(pred_fns)!=1:
            tr()
        else:  pred_fn=pred_fns[0]

        img_fns = [fn for fn in imgs if match_filenames(gt_fn.name,fn.name)==True]

        if len(img_fns)!=1:
            tr()
        else:  img_fn=img_fns[0]
        S = Scorer(img_fn, gt_fn, pred_fn)
        S.process(debug)
        dfs.append(S.df_final)

# %%
    df_final = pd.concat(dfs)
    df_final.to_csv("results/results_16.11.csv",index=False)
# %%

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
    info_from_filename(mask_fn.name)
# %%
    S = Scorer(img_fn, gt_fn, pred_fn)
    S.compute_stats()
    S.compute_overlap_perlesion()
    S.compute_overlap_overall()
    S.cont_tables()
    S.gt_radiomics(debug)
# %%
    S.LG.mask
    view_sitk(S.gt_cc,S.LG.mask)
   
# %%

    LP=S.LP
    
# %%
    gt = S.gt
    mask = sitk.GetArrayFromImage(L.mask)

    view_sitk(S.gt,S.gt_cc)
# %%

    lab_inds = list(il.product(S.labs_gt, S.labs_pred))
# %%
    S.LG = LabelsGeometryAndRadiomics(S.gt_cc)
    S.LG.dust(S.dusting_threshold) # sanity check
    S.gt_cc_dusted = S.LG.mask
    S.LP = LabelsGeometryAndRadiomics(S.pred_cc)
    S.LP.dust(S.dusting_threshold)
    S.pred_cc_dusted = S.LP.mask

    S.labs_pred = S.LP.labels
    S.labs_gt = S.LG.labels


# %%
    L=S.LG

# %%
    dusting_threshold= S.dusting_threshold
    inds_small = [l < dusting_threshold for l in L.lengths.values()]
    L.labels_small = list(il.compress(L.labels, inds_small))
    L._remove_labels(L.labels_small)


# %%
    df = pd.DataFrame(S.dic)
    df["detected"] = S.detected
    names_prefix = lambda jd, seq: [
        "_".join([jd, "pred", "label", str(i)]) for i in seq
    ]
    jac_labels = names_prefix("jac", S.labs_pred)
    dsc_labels = names_prefix("dsc", S.labs_pred)
    jac_df = pd.DataFrame(data=S.jac, columns=jac_labels)
    dsc_df = pd.DataFrame(data=S.dsc, columns=dsc_labels)
    S.df_final["fp_pred_labels"] = [
        S.fp_pred_labels,
    ] * np.maximum(1,len(S.labs_gt))
    S.df_final = pd.concat([df, dsc_df, jac_df], axis=1)
    S.df_final['dsc_overall'], S.df_final['jac_overall'] = S.dsc_overall,S.jac_overall

# %%

    df['fp_pred_labels']= S.fp_pred_labels
# %%
    S2 = Scorer(img_fn, gt_fn, pred_fn)
    S2.compute_stats()
    S2.compute_overlap_perlesion()
    S2.compute_overlap_overall()
    S2.cont_tables()
    S2.gt_radiomics(debug)
    S2.create_df()
# %%
    df = pd.DataFrame(S2.dic)
    df["detected"] = S2.detected
    names_prefix = lambda jd, seq: [
        "_".join([jd, "pred", "label", str(i)]) for i in seq
    ]
    jac_labels = names_prefix("jac", S2.labs_pred)
    dsc_labels = names_prefix("dsc", S2.labs_pred)
    jac_df = pd.DataFrame(data=S2.jac, columns=jac_labels)
    dsc_df = pd.DataFrame(data=S2.dsc, columns=dsc_labels)
    S2.df_final = pd.concat([df, dsc_df, jac_df], axis=1)
    S2.df_final["fp_pred_labels"] = [
        S2.fp_pred_labels,
    ] * np.maximum(1,len(S2.labs_gt))

    S2.df_final['dsc_overall'], S2.df_final['jac_overall'] = S2.dsc_overall,S2.jac_overall

# %%
    df = pd.DataFrame(S.radiomics)
    df["detected"] = S.detected
    names_prefix = lambda jd, seq: [
        "_".join([jd, "pred", "label", str(i)]) for i in seq
    ]
    jac_labels = names_prefix("jac", S.labs_pred)
    dsc_labels = names_prefix("dsc", S.labs_pred)
    jac_df = pd.DataFrame(data=S.jac, columns=jac_labels)
    dsc_df = pd.DataFrame(data=S.dsc, columns=dsc_labels)
    S.df_final = pd.concat([df, dsc_df, jac_df], axis=1)

    S.df_final["fp_pred_labels"] = [
        S.fp_pred_labels,
    ] * np.maximum(1,len(S.labs_gt))

    S.df_final['dsc_overall'], S.df_final['jac_overall'] = S.dsc_overall,S.jac_overall

# %%
    n_predicted = np.sum(S.dsc> S.detection_threshold, 1)
    if n_predicted.size>0:
        S.detected = n_predicted > 0
    else:
        S.detected=None
    tt = S.dsc<= S.detection_threshold
    S.fp_pred_labels = list(np.where(np.all(tt == True, 0))[0] + 1)



