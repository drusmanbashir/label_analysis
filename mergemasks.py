
# %%
from fastai.vision.augment import GetAttr, store_attr
import pandas as pd
import itertools as il
import functools as fl
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses
import six
from fran.utils.fileio import maybe_makedirs
from helpers import *
from radiomics import featureextractor
from pathlib import Path
from fran.transforms.totensor import ToTensorT
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.sitk_utils import align_sitk_imgs
from radiomics_analysis import LesionStats

# %%

# %%
class MergeMasks():
    def __init__(self,fn_label1,fn_label2,output_fname): 
        '''
        used when AI generates organ mask stored in fn_label1. User has drawn lesion masks (fn_label2). This algo will merge masks.

        fn_label1 : Provides label 1.  All others will be erased and holes filled
        fn_label2: Provides label 2.   All others will be erased and holes filled
        '''
        store_attr()
        self.load_images()

    def process(self):
        self.fix_lab1()
        self.fix_lab2()
        self.merge()
        self.write_output()
    def load_images(self):
        self.lab1 = sitk.ReadImage(self.fn_label1)
        self.lab2 = sitk.ReadImage(self.fn_label2)
    def fix_lab2(self):
        self.lab1 
        self.lab2 = sitk.Cast(self.lab2,sitk.sitkLabelUInt16)
        self.lab2 = sitk.ChangeLabelLabelMap(self.lab2,{2:1}) # merge predicted lesions in to organ so there are no holes left.j
        self.lab2 = to_int(self.lab2)
        self.lab2  = sitk.BinaryFillhole(self.lab2)

    def fix_lab1(self):
        self.lab1 = to_int(self.lab1)
        self.lab1  = sitk.BinaryFillhole(self.lab1)
    def merge(self):
        lab1_ar = sitk.GetArrayFromImage(self.lab1)
        lab2_ar = sitk.GetArrayFromImage(self.lab2)
        lab2_ar[lab1_ar==1]=2

        self.lab2_full =sitk.GetImageFromArray(lab2_ar)
        self.lab2_full = align_sitk_imgs(self.lab2_full,self.lab1)

    def write_output(self):
        print("Writing file {}".format(self.output_fname))
        sitk.WriteImage(self.lab2_full,self.output_fname)

# %%
    def merge_multiprocessor(fn_label1,fn_label2,output_fldr,overwrite=False):
        output_fname = output_fldr/fn_label1.name
        if not output_fname.exists() or overwrite==True:
            M = MergeMasks(fn_label1,fn_label2,output_fname)
            M.process()
        else:
            print("File {} exists. Skipping..".format(output_fname))

# %%
#
# fname1 =  Path('/s/datasets_bkp/litqsmall/sitk/masks/litqsmall_00000.nrrd')
# fname2 = Path( '/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357_LITS-413/litqsmall_00000.nrrd')
# merge_multiprocessor(fname1,fname2,output_fldr,overwrite=True)
# %%
if __name__ == "__main__":
    # lesion_masks_folder = Path('/s/datasets_bkp/litqsmall/sitk/masks/')
    lesion_masks_folder = Path('/media/ub2/datasets/drli/sitktmp/masks')
    predicted_masks_folder =Path("/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/") 
    fnames1 = list(lesion_masks_folder.glob("*"))
    fnames_pred = list(predicted_masks_folder.glob("*"))

# %%
    fnames2 = []
    for f in fnames1:
        f2= [fn for fn in fnames_pred if fn.name.split('.')[0] == f.name.split('.')[0]][0]
        fnames2.append(f2)
    
# %%
    output_fldr =lesion_masks_folder.parent/("masks_fused")


    maybe_makedirs(output_fldr)
    args = [[f1,f2,output_fldr, True] for f1,f2 in zip(fnames1,fnames2)]
    multiprocess_multiarg(merge_multiprocessor,args)
# %%
M = MergeMasks(fnames1[0],fnames2[0],output_fldr)
M.fix_lab1()
M.fix_lab2()
# %%
        lab1_ar = sitk.GetArrayFromImage(M.lab1)
        lab2_ar = sitk.GetArrayFromImage(M.lab2)
        lab2_ar[lab1_ar==1]=2

        M.lab2_full =sitk.GetImageFromArray(lab2_ar)
        M.lab2_full = align_sitk_imgs(M.lab2_full,M.lab1)


# %%
    view_sitk(M.lab1,M.lab2,data_types=['mask','mask'])
    M.lab1.GetDirection()
    M.lab2.GetDirection()
# %%
    Mm = sitk.MergeLabelMapFilter
