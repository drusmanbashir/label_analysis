# %%
import shutil
import functools as fl
import pandas as pd
from fastcore.basics import GetAttr, store_attr
import SimpleITK as sitk
from label_analysis import remap
from label_analysis.overlap import LabelMapGeometry, ScorerLabelMaps, get_1lbl_nbrhoods, get_all_nbrhoods, labels_overlap, proximity_indices
from fran.utils.fileio import is_filename, maybe_makedirs
from label_analysis.helpers import *
from pathlib import Path
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from label_analysis.utils import align_sitk_imgs, distance_tuples, is_sitk_file
from typing import Union

from fran.utils.string import find_file, match_filenames, strip_extension


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


class MergeTouchingLabels(LabelMapGeometry):
    '''
    in multiclass UNet output some lesions have areas classified as one class (e.g., benign) and other neighbouring  voxels as malignant, which is nmot possible in real life.
    This algorithm convert 'non-dominant' (e.g. benign) voxels into 'dominant' class voxels where there is overlap
    '''
    def __init__(self, lm,dom_label="larger",ignore_labels=[],threshold=0.1) -> None:

        '''
        threshold: 0.1 if the dominant label has less than thresholded volume contribution, it will be ignored and next largest will be selected instead .
        Whenever more than one labels are presented inside a single cc, the dom_label is assigned to all
        '''
        super().__init__(lm,ignore_labels)
        self.lm_bin_cc, self.nbr_binary= get_1lbl_nbrhoods(self.lm_binary, 1)
        store_attr()


    def process(self):
        self.init_overlap_matrix()
        self.separate_touching_gps()
        self.rank_closest_binary_cc()
        if self.touching_labels==True:
            self.ccs_sharing_binaryblobs()
            self.compute_dsc_touching_labels()
            self.find_dom_labels()
            remapping = self.create_remappings()
            self.lm_out = relabel(self.lm_cc,remapping)
            return self.lm_out
        else:
            return None


    def init_overlap_matrix(self):
            self.dsc = np.zeros(
            [max(1, len(self.nbrhoods)), max(1, len(self.nbr_binary))]
        )


    def separate_touching_gps(self):
        self.nbrhoods['cc_group']= -1
        self.nbr_binary['cc_group']= -1
        self.ccs=self.nbrhoods[~self.nbrhoods.cent.isin(self.nbr_binary.cent)]
        self.ccs['contrib']=0.0
        self.ccs_no_change=self.nbrhoods[self.nbrhoods.cent.isin(self.nbr_binary.cent)]



    def rank_closest_binary_cc(self):
        bins =self.nbr_binary[~self.nbr_binary.cent.isin(self.nbrhoods.cent)]
        distance_matrix = np.zeros((len(bins),len(self.ccs)))
        if len(distance_matrix)>0:
            self.touching_labels=True
            for ind in range(len(self.ccs)):
                cc1 = self.ccs.iloc[ind]
                cent =cc1.cent
                distance_matrix[:,ind] = bins['cent'].apply(lambda x: distance_tuples(x,cent))
                ranking = np.argmin(distance_matrix,axis=0)
                self.ccs = self.ccs.assign(cc_group=ranking)

        else:
            self.touching_labels=False



        
    def ccs_sharing_binaryblobs(self):
        '''
        creates: label_pairs tuples of [cc_label,bin_label]
        '''
        
        bins  = self.nbr_binary[~self.nbr_binary.cent.isin(self.nbrhoods.cent)]
        self.label_pairs=[]
        for ind in range(len(self.ccs)):
            lab_cc = self.ccs.iloc[ind]['label_cc']
            cc_group= self.ccs.iloc[ind]['cc_group']
            lab_bin_cc = bins.iloc[cc_group]['label_cc']
            self.label_pairs.append((int(lab_cc),int(lab_bin_cc)))


    def compute_dsc_touching_labels(self):
        '''
        All ccs sharing same binary blob. Each ccs dsc wrt the binary blob is computed
        
        '''
       
        args = [[self.lm_cc, self.lm_bin_cc, *a, False] for a in self.label_pairs]
        self.dsc = multiprocess_multiarg(
            labels_overlap, args, 16, False, False, progress_bar=True
        )  # multiprocess i s slow
        for ind in range(len(self.label_pairs)):
            label_cc = self.label_pairs[ind][0]
            contrib = self.dsc[ind]
            self.ccs.loc[self.ccs.label_cc==label_cc, 'contrib']= contrib
        assert all(self.ccs.contrib>0), "Some labels are matched to the wrong binary blob and have 0 contrib their matched group which should not possible"


    def find_dom_labels(self):
        cc_groups = self.ccs.cc_group.unique()
        for cc_group in cc_groups:
            ccs_mini = self.ccs[self.ccs.cc_group==cc_group]
            ccs_mini = ccs_mini.loc[ccs_mini['contrib']>self.threshold]
            dom_label = ccs_mini.label.max()
            self.ccs.loc[self.ccs.cc_group==cc_group,'label_out' ]= dom_label
        self.ccs_no_change = self.ccs_no_change.assign(label_out=self.ccs_no_change.label)
        self.nbrhoods_with_remapping  = pd.concat([self.ccs,self.ccs_no_change])

    def create_remappings(self):
        remapping={}
        for i,row in self.nbrhoods_with_remapping.iterrows():
            rem= {int(row['label_cc']):int(row['label_out']) }
            remapping.update(rem)
        return remapping
        


class MergeTouchingLabelsFiles():


        def process_batch(self,lm_fns,output_folder=None, overwrite=False):

            lm_fns,lm_fns_out = self.filter_files(lm_fns,output_folder,overwrite)

            lms = self.load_images(lm_fns)

            lms_fixed = []
            for i, lm in enumerate(lms) :
                    lm_fixed = self.process_lm(lm)
                    lms_fixed.append(lm_fixed)
            for i,lm_fixed in enumerate(lms_fixed):
                    lm_fn_out = lm_fns_out[i]
                    lm_fn = lm_fns[i]
                    if lm_fixed:
                        self.save_fixed_map(lm_fixed,lm_fn_out)
                    else:
                        print("No touching labels found. Manually copying file to output folder")
                        shutil.copy(lm_fn,lm_fn_out)

        def load_images(self,lm_fns):
            lms=[]
            for lm_fn in lm_fns:
                print("Processing {}".format(lm_fn))
                lm = sitk.ReadImage(lm_fn)
                lms.append(lm)
            return lms
        def process_lm(self,lm):
                    Merger = MergeTouchingLabels(lm)
                    lm_fixed = Merger.process()
                    return lm_fixed
            
        @classmethod
        def _from_folder(cls,lm_fldr,output_folder=None):
            lm_fldr = Path(lm_fldr)
            lm_fns = list(lm_fldr.glob("*"))
            return cls(lm_fns,output_folder)



        def filter_files(self,lm_fns,output_folder,overwrite):
            lm_fns = [fn for fn in lm_fns if is_sitk_file(fn)]
            if output_folder is None: self.set_output_folder(lm_fns[0])
            else: self.output_folder = Path(output_folder)
            maybe_makedirs(self.output_folder)
            lm_fns_out = self.set_output_names(lm_fns)
            if overwrite==False:
                for fn, fn_out in zip(lm_fns, lm_fns_out):
                    if fn_out.exists() :
                            print("File {} already exists. Skipping".format(fn))
                            lm_fns.remove(fn)
                            lm_fns_out.remove(fn_out)
            return lm_fns,lm_fns_out


        def set_output_names(self,lm_fns):
            lm_fns_out=[]
            for fn in lm_fns:
                lm_fn_out = self.set_output_name(fn)
                lm_fns_out.append(lm_fn_out)
            return lm_fns_out


        def set_output_name(self, lm_fn):
            lm_fn_out = self.output_folder/(lm_fn.name)
            return lm_fn_out

        def save_fixed_map(self,lm_mod,lm_fn_out):
            lm_mod = to_int(lm_mod)
            maybe_makedirs(self.output_folder)
            sitk.WriteImage(lm_mod,lm_fn_out)
            print("Saved to ",lm_fn_out )


        def set_output_folder(self,lm_fn):
            lm_fn = Path(lm_fn)
            input_fldr = lm_fn.parent
            output_folder_prnt  = input_fldr.parent 
            output_folder_nm =input_fldr.name+"_fixed_mc"
            self.output_folder = output_folder_prnt/output_folder_nm


def merger_wrapper(fns):
    M = MergeTouchingLabelsFiles()
    M.process_batch(fns)

def merge_multiprocessor(lm_fns,overwrite=False,n_chunks=12):
            argsi = list(chunks(lm_fns,n_chunks))
            argsi = [[a] for a in argsi]
            multiprocess_multiarg(merger_wrapper,argsi)



           #
# fname1 =  Path('/s/datasets_bkp/litqsmall/sitk/masks/litqsmall_00000.nrrd')
# fname2 = Path( '/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357_LITS-413/litqsmall_00000.nrrd')
# merge_multiprocessor(fname1,fname2,output_fldr,overwrite=True)
# %%
if __name__ == "__main__":




    preds_fldr = Path("/s/fran_storage/predictions/lidc2/LITS-911")
    lm_fns = list(preds_fldr.glob("*"))
    # M = MergeTouchingLabelsFiles()
    # M.process_batch(lm_fns)
    # lesion_masks_folder = Path('/s/datasets_bkp/litqsmall/sitk/masks/')
    merge_multiprocessor(lm_fns, 6)
# %%

