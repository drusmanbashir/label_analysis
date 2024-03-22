# %%
import shutil
from fastcore.basics import GetAttr, store_attr
import SimpleITK as sitk
from label_analysis.overlap import LabelMapGeometry, get_1lbl_nbrhoods, get_all_nbrhoods
from fran.utils.fileio import maybe_makedirs
from label_analysis.helpers import *
from pathlib import Path
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from label_analysis.utils import align_sitk_imgs
from typing import Union

from fran.utils.string import find_file, match_filenames


class MergeLabelMaps():
    def __init__(self,lab1:Union[Path,sitk.Image],lab2:Union[Path,sitk.Image],output_fname,k_label1=1,debug=True): 
        '''
        used when AI generates organ mask stored in fn_label1. User has drawn lesion masks (fn_label2). This algo will merge masks.

        fn_label1 : Provides label 1 (organ).  All others will be erased and holes filled
        fn_label2: Provides label 2  onwards. 
        debug: breakpoint activates if labelmap2 has a label 1.
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

        lab1_ar = sitk.GetArrayFromImage(self.lab1)
        lab2_ar = sitk.GetArrayFromImage(self.lab2)
        lab2_labels = np.unique(lab2_ar)
        lab2_labels = np.delete(lab2_labels,0)
        for label in lab2_labels:
            if label==1 and self.debug==True:
                tr()
            lab1_ar[lab2_ar==label]=label


        self.lab_merged =sitk.GetImageFromArray(lab1_ar)
        self.lab_merged = align_sitk_imgs(self.lab_merged,self.lab2)
    def write_output(self):
        print("Writing file {}".format(self.output_fname))
        sitk.WriteImage(self.lab_merged,self.output_fname)


def merge_multiprocessor(fn_label1,fn_label2,output_fldr,overwrite=False):
    output_fname = output_fldr/fn_label1.name
    if not output_fname.exists() or overwrite==True:
        M = MergeLabelMaps(fn_label1,fn_label2,output_fname)
        M.process()
    else:
        print("File {} exists. Skipping..".format(output_fname))


class FixMulticlass_CC(LabelMapGeometry):
    '''
    Note: LabelMapGeometry by default ignores label 1. So this algo strips label 1 from every lm and saves it
    in multiclass UNet output some lesions have areas classified as one class (e.g., benign) and other neighbouring  voxels as malignant, which is nmot possible in real life.
    This algorithm convert 'non-dominant' (e.g. benign) voxels into 'dominant' class voxels where there is overlap
    '''

    _default = "fil"

    def __init__(self, lm_fn,dom_label,overwrite=False) -> None:
        '''
        Whenever more than one labels are presented inside a single cc, the dom_label is assigned to all
        '''
        
        store_attr('dom_label')
        self.lm_fn = Path(lm_fn)
        self.set_output_name()
        if overwrite==False and self.lm_fn_out.exists():
            print("output file already exists, skipping")
        else:
            lm = sitk.ReadImage(self.lm_fn)
            super().__init__(lm=lm, ignore_labels=[1])
            _, self.nbr_binary= get_1lbl_nbrhoods(self.lm_binary, 1)
            self.process()

    def process(self):
        if not self.is_empty():
            self.fix_labelmap()
            self.save_fixed_map()
        else:
            print("No lesions. Saving labelmap (excluding label 1, i.e., organ) to",self.lm_fn_out)
            lm_no_organ = to_int(self.lm_org)
            sitk.WriteImage(lm_no_organ,self.lm_fn_out)


    def set_output_name(self):
        output_folder_prnt  = self.lm_fn.parent.parent
        output_folder_nm = self.lm_fn.parent.name+"_fixed_mc"
        output_folder = output_folder_prnt/output_folder_nm
        maybe_makedirs(output_folder)
        self.lm_fn_out = output_folder/(self.lm_fn.name)

    def fix_labelmap(self):
        self.nbrhoods.loc[~self.nbrhoods.cent.isin(self.nbr_binary.cent),'label']=self.dom_label
        remapping  = {}
        for row in self.nbrhoods.iterrows():
            try:
                label_cc , label = row[1].label_cc.item(),  row[1].label.item()
            except:
                label_cc , label = row[1].label_cc,  row[1].label
            remapping.update({label_cc:label})
        self.lm_cc = to_label(self.lm_cc)
        self.lm_mod = sitk.ChangeLabelLabelMap(self.lm_cc,remapping)

    def save_fixed_map(self):
        self.lm_mod = to_int(self.lm_mod)
        try:
            sitk.WriteImage(self.lm_mod,self.lm_fn_out)
            print("Saved to ",self.lm_fn_out )
        except RuntimeError as e:
            print(e)
            print("Saving copy of the original file to: {0}".format(self.lm_fn_out))
            shutil.copy(self.lm_fn,self.lm_fn_out)



#
# fname1 =  Path('/s/datasets_bkp/litqsmall/sitk/masks/litqsmall_00000.nrrd')
# fname2 = Path( '/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357_LITS-413/litqsmall_00000.nrrd')
# merge_multiprocessor(fname1,fname2,output_fldr,overwrite=True)
# %%
if __name__ == "__main__":




    # lesion_masks_folder = Path('/s/datasets_bkp/litqsmall/sitk/masks/')
    lesion_masks_folder = Path('/s/xnat_shadow/crc/srn/cases_with_findings/masks_final/')
    fnames_lab2 = list(lesion_masks_folder.glob("*"))
    fnames_pred = list(predicted_masks_folder.glob("*"))

    output_fldr =lesion_masks_folder
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
    exc_pat = "_\d\.nii"
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-787")
    preds_fldr =Path("/s/fran_storage/predictions/litsmc/LITS-787_LITS-810_LITS-811") 
    for fn in preds_fldr.glob("*"):
    # fn = find_file("CRC018",preds_fldr)

        if fn.is_dir() or re.search(exc_pat,fn.name):
            print("Skipping ",fn)
        else:
            F  = FixMulticlass_CC(fn,3,overwrite=False)
# %%

    lm = sitk.ReadImage(fn)
    L = LabelMapGeometry(lm)

# %%

    M = MergeLabelMaps(fn,fn,output_fldr/("litqsmall_00004_.nrrd"))
    M.process()
# %%
    debug=False
    maybe_makedirs(output_fldr)
    args = [[f1,f2,output_fldr, True] for f1,f2 in zip(fnames_lab1, fnames_lab2)]
    multiprocess_multiarg(merge_multiprocessor,args,debug=debug)
# %%


