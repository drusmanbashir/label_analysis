# %%
import logging
import os
import sys
import time
from functools import reduce

from fran.managers.project import DS
from utilz.cprint import cprint

from label_analysis.geometry import LabelMapGeometry
from label_analysis.geometry_itk import LabelMapGeometryITK
from label_analysis.helpers import remap_single_label
from label_analysis.overlap import chunks
from label_analysis.radiomics_setup import *

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import SimpleITK as sitk
from fastcore.basics import GetAttr
from utilz.helpers import *

from label_analysis.helpers import *

import itk
itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(8)
if not ray.is_initialized():
    ray.init()



def _concat_valid_frames(frames):
    valid_frames = []
    for frame in frames:
        if frame is None or not isinstance(frame, pd.DataFrame):
            continue
        if frame.empty:
            continue
        if frame.dropna(axis=1, how="all").empty:
            continue
        valid_frames.append(frame)

    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


def _processing_error_row(gt_fn, err):
    return pd.DataFrame(
        [
            {
                "lm_filename": gt_fn,
                "processing_error": True,
                "error_type": type(err).__name__,
                "error_message": str(err),
            }
        ]
    )


def _process_labelmap_batch_itk(
    gt_fns,
    ignore_labels=None,
    dusting_threshold=0,
    img_fns=None,
    do_radiomics=True,
    params_fn=None,
):
    nbrhoods = []
    if ignore_labels is None:
        ignore_labels = []
    if isinstance(gt_fns, (str, Path)):
        gt_path = Path(gt_fns)
        gt_fns = list(gt_path.glob("*")) if gt_path.is_dir() else [gt_path]
    elif not isinstance(gt_fns, list):
        gt_fns = list(gt_fns)
    if img_fns is None:
        img_fns = []
    elif isinstance(img_fns, (str, Path)):
        img_path = Path(img_fns)
        img_fns = list(img_path.glob("*")) if img_path.is_dir() else [img_path]
    elif not isinstance(img_fns, list):
        img_fns = list(img_fns)
    if len(img_fns) == 0 and do_radiomics:
        cprint(
            "Warning: no img_files given. Radiomics will not be computed",
            color="red",
            bold=True,
        )
        do_radiomics = False
    for gt_fn in gt_fns:
        if len(img_fns) > 0:
            img_fn = find_matching_fn(
                gt_fn, img_fns, tags=["case_id"], allow_multiple_matches=False
            )
            img = sitk.ReadImage(img_fn)
        else:
            img = None
        try:
            L = LabelMapGeometryITK(li=gt_fn, ignore_labels=ignore_labels, img=img)
            if do_radiomics is True and L.is_empty() is False:
                L.dust(dusting_threshold=dusting_threshold)
                L.radiomics(params_fn)
            L.nbrhoods["lm_filename"] = gt_fn
            L.nbrhoods["processing_error"] = False
            L.nbrhoods["error_type"] = None
            L.nbrhoods["error_message"] = None
            nbrhoods.append(L.nbrhoods)
        except Exception as e:
            logging.exception("Failed processing labelmap file: %s", gt_fn)
            nbrhoods.append(_processing_error_row(gt_fn, e))
    return _concat_valid_frames(nbrhoods)


@ray.remote(num_cpus=8)
class LabelMapGeometryRay:
    def __init__(self):
        pass

    def process(
        self,
        gt_fns,
        ignore_labels=None,
        dusting_threshold=0,
        img_fns=None,
        do_radiomics=True,
        params_fn=None,
    ):
        nbrhoods= []
        if ignore_labels is None:
            ignore_labels = []
        if isinstance(gt_fns, (str, Path)):
            gt_path = Path(gt_fns)
            gt_fns = list(gt_path.glob("*")) if gt_path.is_dir() else [gt_path]
        elif not isinstance(gt_fns, list):
            gt_fns = list(gt_fns)
        if img_fns is None:
            img_fns = []
        elif isinstance(img_fns, (str, Path)):
            img_path = Path(img_fns)
            img_fns = list(img_path.glob("*")) if img_path.is_dir() else [img_path]
        elif not isinstance(img_fns, list):
            img_fns = list(img_fns)
        if len(img_fns) == 0 and do_radiomics:
            cprint("Warning: no img_files given. Radiomics will not be computed",color="red", bold=True)
            do_radiomics=False
        for gt_fn in gt_fns:
            try:
                if len(img_fns)>0:
                    img_fn = find_matching_fn(gt_fn, img_fns,tags=['case_id'],allow_multiple_matches=False)
                    img = sitk.ReadImage(img_fn)
                else:
                    img=None
                L = LabelMapGeometry(lm=gt_fn, ignore_labels=ignore_labels, img=img)
                if do_radiomics==True and L.is_empty()==False:
                    L.dust(dusting_threshold=dusting_threshold  )
                    L.radiomics(params_fn)
                L.nbrhoods["lm_filename"] = gt_fn
                L.nbrhoods["processing_error"] = False
                L.nbrhoods["error_type"] = None
                L.nbrhoods["error_message"] = None
                nbrhoods.append(L.nbrhoods)
            except Exception as e:
                logging.exception("Failed processing labelmap file: %s", gt_fn)
                nbrhoods.append(_processing_error_row(gt_fn, e))

        return _concat_valid_frames(nbrhoods)


@ray.remote(num_cpus=8)
class LabelMapGeometryRayITK:
    def __init__(self):
        pass

    def process(
        self,
        gt_fns,
        ignore_labels=None,
        dusting_threshold=0,
        img_fns=None,
        do_radiomics=True,
        params_fn=None,
    ):
        return _process_labelmap_batch_itk(
            gt_fns=gt_fns,
            ignore_labels=ignore_labels,
            dusting_threshold=dusting_threshold,
            img_fns=img_fns,
            do_radiomics=do_radiomics,
            params_fn=params_fn,
        )

@ray.remote(num_cpus=8)
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

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":
# %%
    preds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-940_fixed_mc"
    )
    test_lms_fldr = Path("/s/xnat_shadow/crc/lms")
    train_lms_fldrs= [Path(fldr)/("lms") for fldr in train_fldrs]
    train_img_fldrs = [Path(fldr)/("images") for fldr in train_fldrs]
    train_img_fns =[list(fldr.glob("*")) for fldr in train_img_fldrs]
    train_img_fns = list(il.chain.from_iterable(train_img_fns))
    train_lm_fns = [list(fldr.glob("*")) for fldr in train_lms_fldrs]
    train_lm_fns = list(il.chain.from_iterable(train_lm_fns))

    test_imgs_fldr = Path("/s/xnat_shadow/crc/images")
    pred_fns = list(preds_fldr.glob("*"))
    test_lm_fns =  list(test_lms_fldr.glob("*"))
    test_img_fns =  list(test_imgs_fldr.glob("*"))

    lms_pt_fldr = Path("/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/lms")
    lms_pt = list(lms_pt_fldr.glob("*"))
    lm_fn = lms_pt[0]

    lm = torch.load(lm_fn,weights_only=False)
    lm.meta
# %%
    DS.lidc
    lms_lidc = DS.lidc.folder/("lms")
    fns_lidc = list(lms_lidc.glob("*"))
# %%
#SECTION:-------------------- Multiprocess Training data NO RADIOMICS--------------------------------------------------------------------------------------
    
    fldr_pt = Path("/r/datasets/preprocessed/lidc/lbd/spc_075_075_075_rlb109adb5e_rlb109adb5e_ex000/lms")
    fns_pt = list(fldr_pt.glob("*"))
    fns_pt = ["/media/UB/datasets/kits21/lms/kits21_00018.nii.gz"]


    n_actors = 1
    fns_chunks = list(chunks(fns_pt, n_actors))
    actors = [LabelMapGeometryRay.remote() for _ in range(n_actors)]

# %%
    ignore_labels = [1]
    do_radiomics = False
    params_fn = None
    dusting_threshold = 1

# %%
    futures = []
    for actor, fns_chunk in zip(actors, fns_chunks):
        res = actor.process.remote(
            fns_chunk,
            ignore_labels,
            dusting_threshold,
            img_fns=None,
            do_radiomics=do_radiomics,
            params_fn=params_fn,
        )
        futures.append(res)

    parts = ray.get(futures)
# %%
    resdf = pd.concat(parts, ignore_index=True)
    out_fn = fldr_pt.parent/("lesion_stats.csv")

    resdf['case_id'] = resdf['lm_filename'].apply(split_info)
    resdf.to_csv( out_fn, index=False)

    df = resdf.groupby("case_id")["volume"].apply(list)

    case_ids= resdf["case_id"].unique()

    grouped = df.copy()
# %%
    vol_lists = grouped.tolist()

    max_len = max(len(v) for v in vol_lists)

# pad lists so matrix can be built
    vol_matrix = []
    for v in vol_lists:
        padded = v + [0]*(max_len-len(v))
        vol_matrix.append(padded)

    vol_matrix = list(zip(*vol_matrix))

# %%
    import seaborn as sns
    import math     
    chunk_size = 50
    n_chunks = math.ceil(len(case_ids) / chunk_size)


# %%
