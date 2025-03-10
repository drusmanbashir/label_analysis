# %%
import sys
from radiomics import featureextractor

import ray

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr
from label_analysis.helpers import *

from utilz.fileio import maybe_makedirs
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.string import (
    find_file,
    info_from_filename,
    match_filenames,
    strip_extension,
    strip_slicer_strings,
)
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})



def do_radiomics(img: sitk.Image, lm: sitk.Image, label: int, mask_fn=None, paramsFile=None):
    if not paramsFile:
        paramsFile = "label_analysis/configs/params.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)

    featureVector = {}
    featureVector["case_id"] = info_from_filename(mask_fn.name)["case_id"]
    featureVector["fn"] = mask_fn
    featureVector2 = extractor.execute(img, lm, label=label)
    featureVector["label"] = featureVector2["diagnostics_Configuration_Settings"][
        "label"
    ]
    featureVector.update(featureVector2)
    return featureVector


def radiomics_multiprocess(img, lm_cc, labels, lm_fn, params_fn=None, debug=False):
    print("Computing lm label radiomics")
    args = [[img, lm_cc, label, lm_fn, params_fn] for label in labels]
    radiomics = multiprocess_multiarg(
        do_radiomics,
        args,
        num_processes=8,
        multiprocess=True,
        debug=debug,
    )
    return radiomics


# %%
if __name__ == "__main__":  #


    pass
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

