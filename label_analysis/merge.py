# %%
import shutil
import functools as fl
from fastcore.basics import GetAttr, store_attr
import SimpleITK as sitk
from label_analysis import remap
from label_analysis.overlap import LabelMapGeometry, get_1lbl_nbrhoods, get_all_nbrhoods
from fran.utils.fileio import is_filename, maybe_makedirs
from label_analysis.helpers import *
from pathlib import Path
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from label_analysis.utils import align_sitk_imgs
from typing import Union

from fran.utils.string import find_file, match_filenames


def merge(lm_base, lms_other:Union[sitk.Image,list],labels_lol:list =None):


    #labels_lol:i.e.., list of lists. One set of labels per lm. If not provided, all labels in each lm are used. This is necessary as a uniform template. Some lm may not have full compliment of labels. note lm_base labels are not required
    # those earlier in order get overwritten by those later. So lesions should be last
    if not isinstance(lms_other,Union[tuple,list]): lms_other = [lms_other]
    def _inner(lm2_ar,labels):
        for label in labels:
            lm_base_arr[lm2_ar==label]=label


    lm_base_arr= sitk.GetArrayFromImage(lm_base)
    lm_arrs =[ sitk.GetArrayFromImage(lm) for lm in lms_other]
    if labels_lol is None:
        labels_lol  = [get_labels(lm) for lm in lms_other]

    lm_final = [_inner(lm_arr,labels) for lm_arr,labels in zip (lm_arrs,labels_lol)][0]
    lm_final=sitk.GetImageFromArray(lm_final)
    lm_final= align_sitk_imgs(lm_final,lm_base)
    return lm_final


def merge_pt(lm_base_arr, lm_arrs:Union[sitk.Image,list],labels_lol:list =None):

    #labels_lol:i.e.., list of lists. One set of labels per lm. If not provided, all labels in each lm are used. This is necessary as a uniform template. Some lm may not have full compliment of labels. note lm_base labels are not required
    # those earlier in order get overwritten by those later. So lesions should be last
    if not isinstance(lm_arrs,Union[tuple,list]): lm_arrs = [lm_arrs]
    def _inner(lm2_ar,labels):
        for label in labels:
            lm_base_arr[lm2_ar==label]=label
        return lm_base_arr

    if labels_lol is None:
        labels_lol  = [lm.unique()[1:] for lm in lm_arrs]

    lm_final = [_inner(lm_arr,labels) for lm_arr,labels in zip (lm_arrs,labels_lol)][0]
    return lm_final

def merge_lmfiles(lm_base, lms_other:Union[sitk.Image,list],labels_lol:list =None):

    lm_base = sitk.ReadImage(lm_base)

    if not isinstance(lms_other,Union[tuple,list]): lms_other = [lms_other]
    lms_other = [sitk.ReadImage(lm) for lm in lms_other]
    return merge(lm_base,lms_other)



class MergeLabelMaps():
    def __init__(self,lm_fns:list, labels:list, output_fname:Union[Path,str],remappings:list=None,file_holes:list=None ):
        '''
        all lms must have non-overlapping labels.
        used when AI generates organ mask stored in fn_label1. User has drawn lesion masks (fn_label2). This algo will merge masks.

        fn_label1 : Provides label 1 (organ).  All others will be erased and holes filled
        fn_label2: Provides label 2  onwards. 
        debug: breakpoint activates if labelmap2 has a label 1.
        '''
        if all([isinstance(fn,str) or isinstance(fn,Path) for fn in lm_fns]): # isinstance(fn,Path):
            self.load_images(lm_fns)

    def process(self):
        self.fix_lab1()
        self.fix_lm()
        self.merge()
        self.write_output()
    def load_images(self,lm_fns):
        self.lab1 = sitk.ReadImage(lm_fns[0])
        self.lab2 = sitk.ReadImage(lm_fns[1])
    def fix_lm(self,lm,remapping):
        lm = relabel(lm,remapping)
        self.lab2 = to_int(self.lab2)

        self.lab2  = sitk.BinaryFillhole(self.lab2)

    def fix_lab1(self):
        self.lab1 = to_int(self.lab1)
        self.lab1 = sitk.Cast(self.lab1,sitk.sitkLabelUInt16)
        if self.remapping1:
            self.lab1 = sitk.ChangeLabelLabelMap(self.lab1,{2:1}) # merge predicted lesions in to organ so there are no holes left.j
        self.lab1 = to_int(self.lab1)
        self.lab1  = sitk.BinaryFillhole(self.lab1)
    def merge(self):
        lab1_ar = sitk.GetArrayFromImage(self.lab1)
        lab2_ar = sitk.GetArrayFromImage(self.lab2)
        lab2_labels = np.unique(lab2_ar)
        lab2_labels = np.delete(lab2_labels,0)
        for label in lab2_labels:
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
# %%
    from label_analysis.totalseg import TotalSegmentorLabels 
# %%

    output_fldr=Path("/s/fran_storage/labelmaps_mod/tmp/")

    M = MergeLabelMaps(fn1,fn2,output_fldr/(fn1))
    M.process()
# %%
    debug=False
    maybe_makedirs(output_fldr)
    args = [[f1,f2,output_fldr, True] for f1,f2 in zip(fnames_lab1, fnames_lab2)]
    multiprocess_multiarg(merge_multiprocessor,args,debug=debug)
# %%
# %%


