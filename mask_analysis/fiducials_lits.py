# %%
from pathlib import Path
from pandas import unique
from fran.utils.fileio import load_json
from fran.utils.helpers import find_matching_fn, pp
from fran.utils.imageviewers import view_sitk
from mask_analysis.helpers import astype, get_labels, single_label, to_cc, to_int
import itertools as il
import numpy as np
import SimpleITK as sitk
import ipdb
import pandas as pd

from fran.utils.string import strip_extension
tr = ipdb.set_trace


met_labels = [ 'mets']

# %%
if __name__ == "__main__":

# %% [markdown]
## AIscreening dataset
# %%

 %%
# %%
    lm  = sitk.ReadImage(label_fn)
    lm = sitk.Cast(lm, sitk.sitkUInt8)
    fil = sitk.LabelShapeStatisticsImageFilter()
    if lesion_labels == 'mets':
        remapping = {1:1,2:3}
    lm = sitk.ChangeLabel(lm,remapping)
# %%
    sitk.WriteImage(lm,fn_out)
    view_sitk(lm,lm)
# %%
    flist = label_fns
    markup_fns = [f for f in flist if "mrk.json"  in f.name and digits in f.name]
    markup_fn =markup_fns[0]
    jj = load_json(markup_fn)
    markup_type ='mets' if met_labels[0] in markup_fn.name else 'benign'
    if markup_type=='mets' :
        target_label, non_target_label = 3,2
    else:
        target_label, non_target_label = 2,3
    mu = jj['markups']
    locs = mu[0]['controlPoints']
    locs2 = [a['position'] for a in locs]
    pp(locs2)
    locs2 = np.array(locs2)
# %%
    # assert((len(closest_markers)) == (len(lu:=np.unique(closest_markers)))), "Repeat values Some fiducials are closest to more than one lesion. It may be an orphan or from other category"
    
    print(remapping)
    lm = sitk.ChangeLabel(lm,remapping)
    sitk.WriteImage(lm,fn_out)
# %%

    view_sitk(lm,lm)
# %%
