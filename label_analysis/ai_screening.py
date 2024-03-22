# %%
from functools import reduce
import sys
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import shutil
from label_analysis.merge import MergeLabelMaps

from label_analysis.overlap import BatchScorer, LabelMapGeometry, Scorer
from label_analysis.remap import RemapFromMarkup


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

# %%
if __name__ == "__main__":
    df = pd.read_excel("/s/xnat_shadow/crc/dcm_summary_latest.xlsx")
    df = pd.read_excel("/s/xnat_shadow/crc/wxh/wxh_summary.xlsx")
    lesion_masks_folder = Path('/s/xnat_shadow/crc/wxh/masks_manual_todo')
    lm_final_fldr = Path("/s/xnat_shadow/crc/wxh/masks_manual_final/")
    lesion_masks_folder2 = Path('/s/xnat_shadow/crc/srn/cases_with_findings/masks')
    marksups_fldr = Path("/s/xnat_shadow/crc/wxh/markups/")
    
    fnames_lab = list(lesion_masks_folder.glob("*"))
    fnames_final = list(lm_final_fldr.glob("*"))
    set(fnames_lab).difference(fnames_final)
    output_fldr =lesion_masks_folder.parent/("masks_manual_final")
    maybe_makedirs(output_fldr)
# %%
    fnames_json = list(marksups_fldr.glob("*"))
    for fn_j in fnames_json:
        cid = info_from_filename(fn_j.name)['case_id']
        lm_fn = [fn for fn in fnames_lab if cid in fn.name]

        lm_fn = lm_fn[0]
        lm_fn_out = output_fldr/(lm_fn.name)
        R = RemapFromMarkup(organ_label =None)
        R.process(lm_fn,lm_fn_out,fn_j)

# %%
    # fnames_lab = list(lesion_masks_folder.glob("*"))
# %%
    imgs_fldr = Path("/s/xnat_shadow/crc/wxh/images/")
    imgs  = list(imgs_fldr.glob("*"))
    preds_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed")
    lm_final = list(lm_final_fldr.glob("*"))

    pending = []
    for img_fn in imgs:
        case_id = info_from_filename(img_fn.name)['case_id']
        lm_done = [fn for fn in lm_final if case_id in fn.name]
        if len(lm_done)==0:
            done = False
        else:
            done = True
        pending.append(not done)

# %%
    imgs_pending = list(il.compress(imgs,pending))
    # img = imgs_pending[0]
    # case_id = info_from_filename(img.name)['case_id']
    # case_id = "crc_"+case_id
    # row = df.loc[df.case_id==case_id]
    # colnames = list(df.columns)+["disparity"]
    df2 = pd.DataFrame(columns=df.columns)
    df2['disparity']=0



# %%

    fnames_lab = list(lesion_masks_folder.glob("*"))
    R = RemapFromMarkup(organ_label =None)
    # for idx in range(len(df)):
    for idx,img in enumerate(imgs):
        case_id = info_from_filename(img.name)['case_id']
        case_id = "crc_"+case_id
        row = df.loc[df.case_id==case_id]
    # idx = 0
        # print("---",idx)
        # row = df.loc[idx]
        row2 = row.copy()
        row2['disparity'] = 0
        # row2.columns = colnames
        print(row.labels)

        case_id = row.case_id.item()
        lab = row.labels.item()


        if lab == "exclude" or lab=='done':
            print("Case excluded/ done:  ",case_id)
            pass
        else:
            fn_lm = [fn for fn in fnames_lab if case_id in fn.name]
            if len(fn_lm)!=1: 
                tr()
            else:
                fn_lm =fn_lm[0]
            
            fn_out = output_fldr/(fn_lm.name)

            if fn_out.exists() :
                print("File exists: ",fn_out)
                pass

            else:
                lm = sitk.ReadImage(fn_lm)
                lg= LabelMapGeometry(lm,ignore_labels=[])
                if lab == 'normal' :
                    if len(lg)!=0:
                        row2['disparity'] = 1
                        print(lg.nbrhoods.label)
                        if all(lg.nbrhoods.label == 2):
                            print("Labels are benign")
                            row2.labels ='benign'
                            shutil.copy(fn_lm,fn_out)
                        else:
                            tr()
                            remapper = {1:2}
                            lm = relabel(lm,remapper)
                            sitk.WriteImage(lm,fn_out)
                    else:
                        row2['disparity'] = 0
                        shutil.copy(fn_lm,fn_out)
                    df2.loc[idx]=row2.iloc[0]

                elif lab=='benign':
                    if all(lg.nbrhoods.label == 2):
                        shutil.copy(fn_lm,fn_out)

                    elif all(lg.nbrhoods.label == 1):
                        remapper = {1:2}
                        lm = relabel(lm,remapper)
                        sitk.WriteImage(lm,fn_out)
                    elif len(lg)==0:
                        row2.labels='normal'
                        row2.disparity=1
                        shutil.copy(fn_lm,fn_out)
                    else:

                        tr()

                    df2.loc[idx]=row2.iloc[0]
                elif lab == "done":
                    tr()

                    remapper = {1:2,2:3}
                    lm = relabel(lm,remapper)
                    sitk.WriteImage(lm,fn_out)

                elif lab == 'mets':
                    if all(lg.nbrhoods.label == 3):
                        shutil.copy(fn_lm,fn_out)
                        df2.loc[idx]=row2.iloc[0]
                    elif all(lg.nbrhoods.label == 1):
                        remapper = {1:3}
                        lm = relabel(lm,remapper)
                        sitk.WriteImage(lm,fn_out)

                    else:
                        tr()
                elif lab == 'json' or lab == 'markup':
                    fn_js = [fn for fn in fnames_json if case_id in fn.name]
                    if len(fn_js)==1:
                        R.process(lm_fn,lm_fn_out,fn_js[0])
                    else:
                        tr()


# %%
# %%
    exc_pat = "_\d\.nii"
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_LITS-810_LITS-811_fixed_mc/")
    for fn in preds_fldr.glob("*"):
    # fn = find_file("CRC018",preds_fldr)

        if fn.is_dir() or re.search(exc_pat,fn.name):
            print("Skipping ",fn)
        else:
            F  = FixMulticlass_CC(fn,3,overwrite=True)

# %%
    gt_fldr = Path("/s/xnat_shadow/crc/wxh/masks_manual_final/")
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    gt_fns = list(gt_fldr.glob("*"))



# %%
    do_radiomics=False

# %%
    B = BatchScorer(gt_fns,imgs_fldr=imgs_fldr,preds_fldr=preds_fldr,debug=False,do_radiomics=False)#,output_fldr=Path("/s/fran_storage/predictions/litsmc/LITS-787_mod/results"))
    B.process()
# %%
    df = pd.read_csv(B.output_fn)
    excluded = list(pd.unique(df['gt_fn'].dropna()))
# %%
    cids = np.array([info_from_filename(fn.name)['case_id'] for fn in gt_fns])
    news = []
    dups = []
    for id in cids:
        if id not in news:
            news.append(id)
        else:
            dups.append(id)

# %%
    case_subid = "CRC311"
    gt_fn = find_file(case_subid,gt_fns) 
    pred_fn= find_file(case_subid,preds_fldr)
# %%

    do_radiomics=False
    S = Scorer(gt_fn,pred_fn,img_fn=None,ignore_labels_gt=[],ignore_labels_pred=[1],save_matrices=False,do_radiomics=do_radiomics)
    df = S.process()
# %%

    predicted_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed')
    predicted_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/preds')
    lesion_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/masks_lesions_are_label1')
    lesion_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/masks_no_liver/')
# %%
    mapping = {1:2}
    for lm_fn in lesion_masks_folder.glob("*"):
        lm = sitk.ReadImage(lm_fn)
        lm= to_label(lm)
        lm = sitk.ChangeLabelLabelMap(lm,mapping)
        lm = to_int(lm)
        sitk.WriteImage(lm,lm_fn)


# %%

    mask_fns = list(lesion_masks_folder.glob("*"))
    pred_fns = list(predicted_masks_folder.glob("*"))
# %%
    output_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/masks_final_with_liver/")
    for mask_fn in mask_fns:
        # mask_fn = mask_fns[0]
        output_fn = output_fldr/(mask_fn.name)
        if not output_fn.exists():
            pred_fn = find_matching_fn(mask_fn,pred_fns)
            M = MergeLabelMaps(pred_fn,mask_fn,output_fn)

            M.process()
        else:
            print("File exists: ",output_fn)
# %%
    out_fns=[]
    imgs_fldr = Path("/s/xnat_shadow/crc/srn/cases_with_findings/images_done/")
    img_fns = list(imgs_fldr.glob("*"))
    for mask_fn in output_fldr.glob("*"):
        out_fns.append(find_matching_fn(mask_fn, img_fns))

# %%
    predicted_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/preds_fixed')

    pred_fns = list(predicted_masks_folder.glob("*"))
# %%
    for pred_fn in pred_fns:
        fn_out = output_fldr/(pred_fn.name)
        if fn_out.exists():
            print("File exists skipping",fn_out)
        else:
            case_id = info_from_filename(pred_fn.name)['case_id']
            case_id = "crc_"+case_id
            row = df.loc[df.case_id==case_id]
            lab = row.labels.item()
            if lab == 'benign':
                labels_expected = [1,2]
            elif lab == 'mets':
                labels_expected = [1,3]
            else:
                labels_expected = [1,2,3]

            lm = sitk.ReadImage(pred_fn)
            labels = get_labels(lm)
            if labels==labels_expected: 
                print("files concur",pred_fn)
                shutil.copy(pred_fn,fn_out)
            else:
                remapping = {2:3, 3:2}
                tr()
                lm = relabel(lm,remapping)
                sitk.WriteImage(lm,fn_out)
        
# %%
fn = "/s/xnat_shadow/crc/wxh/masks_manual_final/crc_CRC159_20161122_ABDOMEN-Segment_1-label.nrrd"
lm = sitk.ReadImage(fn)

lg= LabelMapGeometry(lm,ignore_labels=[])
# %%

fnames_lab = list(lesion_masks_folder.glob("*"))
fnames_lab =[ fn.name for fn in fnames_lab]
fnames_final = list(lm_final_fldr.glob("*"))
fnames_final =[ fn.name for fn in fnames_final]
pending_m = list(set(fnames_lab).difference(fnames_final))


pp(pending_m)
# %%
