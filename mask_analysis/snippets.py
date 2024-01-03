
# %%
import sys
from fran.utils.imageviewers import view_sitk, ImageMaskViewer
sys.path+=["/home/ub/code"]
from mask_analysis.helpers import to_int, to_label
import SimpleITK as sitk
from pathlib import Path
import ipdb

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
