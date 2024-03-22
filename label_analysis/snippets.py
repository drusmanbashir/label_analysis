
# %%
import SimpleITK as sitk
from fran.utils.helpers import multiprocess_multiarg
from label_analysis.helpers import remove_organ_label
from label_analysis.overlap import LabelMapGeometry, Scorer
from pathlib import Path
import pandas as pd

import sys
from fran.utils.imageviewers import view_sitk, ImageMaskViewer
from fran.utils.string import find_file
sys.path+=["/home/ub/code"]
from label_analysis.helpers import to_int, to_label
import SimpleITK as sitk
from pathlib import Path
import ipdb

def lesion_stat_wrapper(fn):
        img = sitk.ReadImage(fn)
        img = remove_organ_label(img,tumour_always_present=False)
        L = LabelMapGeometry(img)
        L.dust(3)
        return fn,L.n_labels,L.lengths
    


# %%

# %%
if __name__ == "__main__":

# %%
    preds_fldr = Path(
    "/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
    )


    normals = [fn for fn in preds_fldr.glob("*nrrd") if "normal" in fn.name]
    args = [[fn] for fn in normals]
    output =   multiprocess_multiarg(lesion_stat_wrapper, args)
    df = pd.DataFrame(data=output,columns =['filename','number of lesions','lengths'])
    df.to_csv("~/code/mask_analysis/results/normal_cases_analysis.csv")
    tr = ipdb.set_trace
    imgs=Path("/home/ub/Desktop/capestart/liver/images/")

    inf =Path("/s/fran_storage/predictions/lits/ensemble_LITS-482_LITS-478_LITS-476/")
    outf =Path("/home/ub/Desktop/capestart/liver/masks/") 
    masksfldr=Path("/home/ub/Desktop/capestart/liver/masks/")
    ms=list(inf.glob("*"))
# %%
    m2=[]
    for x in imgs.glob("*"):
        for m in inf.glob("*"):
            if m.name==x.name:
                # tr()
                m2.append(m)

# %%
    for maskfn in m2:
        out_fname = outf/(maskfn.name.replace("nrrd","nii.gz"))
        mask=sitk.ReadImage(maskfn)
        mask=to_label(mask)
        mask2 =  sitk.ChangeLabelLabelMap(mask, {2: 1})
        mm=to_int(mask2)
        sitk.WriteImage(mm,out_fname)


# %%
    m1=sitk.GetArrayFromImage(to_int(mask))
    ImageMaskViewer([m2,m1],data_types=['mask,mask'])
# %%

    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787_mod")

    gt_fldr = Path("/s/xnat_shadow/crc/completed/masks")
    imgs_fldr = Path("/s/xnat_shadow/crc/completed/images")

    gt_fns = list(gt_fldr.glob("*"))



    case_ = "CRC075"
    gt_fn  = find_file(case_,gt_fns)
    pred_fn = find_file(case_,preds_fldr)

# %%
    do_radiomics=False
    S = Scorer(gt_fn,pred_fn,img_fn=None,ignore_labels_gt=[],ignore_labels_pred=[1],save_matrices=False,do_radiomics=do_radiomics)
    df = S.process()
#
