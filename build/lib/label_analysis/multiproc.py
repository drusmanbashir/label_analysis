# %%
import os

from label_analysis.geometry import LabelMapGeometry
from label_analysis.radiomics import *
import logging
import sys
import time
from functools import reduce

from label_analysis.helpers import remap_single_label

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr
from label_analysis.helpers import *

from utilz.helpers import *
ray.init()


@ray.remote(num_cpus=4)
class LabelMapGeometryRay:
    def __init__(self):
        pass

    def process(self, gt_fns, img_fns, ignore_labels,fn_indices, do_radiomics=True,params_fn=None,):
        nbrhoods= []
        if not isinstance(gt_fns, list) and  gt_fns.is_dir():
            gt_fns = list(gt_fns.glob("*"))
        if not isinstance(img_fns, list) and img_fns.is_dir():
            img_fns = list(img_fns.glob("*"))
        for indx in fn_indices:
            gt_fn =  gt_fns[indx]

            img_fn = find_matching_fn(gt_fn, img_fns,True)
            img = sitk.ReadImage(img_fn)
            L = LabelMapGeometry(lm=gt_fn, ignore_labels=ignore_labels, img=img)
            if do_radiomics==True and L.is_empty()==False:
                L.dust(3)
                L.radiomics(params_fn)
            nbrhoods.append(L.nbrhoods)
        dfs = pd.concat(nbrhoods, ignore_index=True)
        return dfs


# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":
# %%
    from fran.managers.datasource import _DS
    DS = _DS()
    preds_fldr = Path(
        "/s/fran_storage/predictions/litsmc/LITS-940_fixed_mc"
    )
    test_lms_fldr = Path("/s/xnat_shadow/crc/lms")
    train_fldrs = DS.lits['folder'], DS.litq['folder'] , DS.drli['folder'], DS.litqsmall['folder']
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

# %%
#SECTION:-------------------- RADIOMICS MUltiprocess Training data--------------------------------------------------------------------------------------
    indices = range(len(train_lm_fns))
    # indices = range(24)
    indices = list(chunks(indices,4))

    ignore_labels=[1]
    actors = [LabelMapGeometryRay.remote() for x in range(4)]

# %%
    do_radiomics = True
    params_fn = None
    res = ray.get([c.process.remote(train_lm_fns,train_img_fns,ignore_labels,inds,do_radiomics,params_fn) for c,inds in zip(actors,indices)])
# %%
    res.to_csv("results/results_training.csv")
    resdf = [pd.DataFrame(r) for r in res]
    resdf = pd.concat(res, ignore_index=True)
# %%

# %%
    fulls =[]
    for fn in test_lm_fns[:24]:
        L = LabelMapGeometry(lm=fn, ignore_labels=[1])
        # L.dust(3)
        if not L.is_empty():
            fulls.append(L)
# %%

