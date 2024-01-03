# %%
from fastcore.basics import GetAttr, store_attr
import SimpleITK as sitk
from fran.utils.fileio import maybe_makedirs
from mask_analysis.helpers import *
from pathlib import Path
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.sitk_utils import align_sitk_imgs

from fran.utils.string import match_filenames


# %%
class MergeMasks():
    def __init__(self,lab1:Union[Path,sitk.Image],lab2:Union[Path,sitk.Image],output_fname,k_label1=1): 
        '''
        used when AI generates organ mask stored in fn_label1. User has drawn lesion masks (fn_label2). This algo will merge masks.

        fn_label1 : Provides label 1 (organ).  All others will be erased and holes filled
        fn_label2: Provides label 2 (lesions).  This label is assumed to be stored as 1. Will be remapped to 2. All others will be erased and holes filled
        k_label1: If the organ predictions have extra particles, every label other than the largest k will be removed
        '''
        store_attr()
        if all([isinstance(fn,str) or isinstance(fn,Path) for fn in [self.lab1,self.lab2]]): # isinstance(fn,Path):
            self.load_images()

    def process(self):
        self.fix_lab1()
        self.fix_lab2()
        self.merge()
        self.write_output()
    def load_images(self):
        self.lab1 = sitk.ReadImage(self.lab1)
        self.lab2 = sitk.ReadImage(self.lab2)
    def fix_lab2(self):
        self.lab2 = to_int(self.lab2)
        self.lab2  = sitk.BinaryFillhole(self.lab2)

    def fix_lab1(self):
        self.lab1 = to_int(self.lab1)
        self.lab1 = sitk.Cast(self.lab1,sitk.sitkLabelUInt16)
        self.lab1 = sitk.ChangeLabelLabelMap(self.lab1,{2:1}) # merge predicted lesions in to organ so there are no holes left.j
        self.lab1 = to_int(self.lab1)
        if self.k_label1==1 :
            cc = sitk.ConnectedComponent(self.lab1)
            cc = sitk.RelabelComponent(cc)
            self.lab1 = single_label(cc,1)
            self.lab1 = to_int(self.lab1)
        else:
            tr()
        self.lab1  = sitk.BinaryFillhole(self.lab1)
    def merge(self):
        lab1_ar = sitk.GetArrayFromImage(self.lab1)
        lab2_ar = sitk.GetArrayFromImage(self.lab2)
        lab1_ar[lab2_ar==1]=2

        self.lab_merged =sitk.GetImageFromArray(lab1_ar)
        self.lab_merged = align_sitk_imgs(self.lab_merged,self.lab2)
    def write_output(self):
        print("Writing file {}".format(self.output_fname))
        sitk.WriteImage(self.lab_merged,self.output_fname)


def merge_multiprocessor(fn_label1,fn_label2,output_fldr,overwrite=False):
    output_fname = output_fldr/fn_label1.name
    if not output_fname.exists() or overwrite==True:
        M = MergeMasks(fn_label1,fn_label2,output_fname)
        M.process()
    else:
        print("File {} exists. Skipping..".format(output_fname))


#
# fname1 =  Path('/s/datasets_bkp/litqsmall/sitk/masks/litqsmall_00000.nrrd')
# fname2 = Path( '/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357_LITS-413/litqsmall_00000.nrrd')
# merge_multiprocessor(fname1,fname2,output_fldr,overwrite=True)
# %%
if __name__ == "__main__":




    # lesion_masks_folder = Path('/s/datasets_bkp/litqsmall/sitk/masks/')
    lesion_masks_folder = Path('/s/xnat_shadow/crc/test/labels')
    predicted_masks_folder =Path("/s/fran_storage/predictions/lits32/LIT-143_LIT-150_LIT-149_LIT-153_LIT-161") 
    fnames_lab2 = list(lesion_masks_folder.glob("*"))
    fnames_pred = list(predicted_masks_folder.glob("*"))

    output_fldr =Path("/s/datasets_bkp/litqsmall/masks")
# %%
    fnames_lab1 = []
    for f in fnames_lab2:
        f2= [fn for fn in fnames_pred if match_filenames(f.name,fn.name)==True]
        if len(f2)!=1:
            tr()
        else:
            f2=f2[0]
        fnames_lab1.append(f2)
    
# %%

    fn = "/s/datasets_bkp/litqsmall/masks/litqsmall_00004 (copy).nrrd"
# %%
    view_sitk(lm,lm2)
    M = MergeMasks(fn,fn,output_fldr/("litqsmall_00004_.nrrd"))
    M.process()
# %%
    debug=False
    maybe_makedirs(output_fldr)
    args = [[f1,f2,output_fldr, True] for f1,f2 in zip(fnames_lab1, fnames_lab2)]
    multiprocess_multiarg(merge_multiprocessor,args,debug=debug)
# %%


