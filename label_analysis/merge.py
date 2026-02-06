# %%
import itertools as il
import shutil
from pathlib import Path
from typing import Union

from label_analysis.utils import align_sitk_imgs
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import store_attr
from label_analysis.helpers import *
from label_analysis.overlap import (LabelMapGeometry, get_1lbl_nbrhoods,
                                    labels_overlap )

from utilz.fileio import maybe_makedirs
from utilz.helpers import *
from utilz.imageviewers import *
#
# def fixed_output_folder(lm_fn):
#             lm_fn = Path(lm_fn)
#             input_fldr = lm_fn.parent
#             output_folder_prnt  = input_fldr.parent 
#             output_folder_nm =input_fldr.name+"_fixed_mc"
#             output_folder = output_folder_prnt/output_folder_nm
#             return output_folder
#
#


class MergeLabelMaps():
    def __init__(self,lm_fn1,lm_fn2 , output_fname:Union[Path,str],remapping1=None,remapping2=None):
        '''
        all lms must have non-overlapping labels.
        used when AI generates organ mask stored in fn_label1. User has drawn lesion masks (fn_label2). This algo will merge masks.

        lm_fns list:
            fn_label1 : Provides label 1 (organ).  #NOTE : All others will be erased and holes filled
            fn_label2: Provides label 2  onwards. 
        remapping: list of dicts , one for each of the pair of lm_fns. 
        debug: breakpoint activates if labelmap2 has a label 1.
        '''

        self.lm1 = sitk.ReadImage(str(lm_fn1))
        self.lm2 = sitk.ReadImage(str(lm_fn2))
        assert(self.lm1.GetSize()==self.lm2.GetSize()),"Size mismatch between labelmaps. \nFile 1: {0},\nFile 2: {1}.\n------------------------------------".format(lm_fn1,lm_fn2)
        store_attr('output_fname,remapping1,remapping2')

    def process(self):
        self.fix_lm1()
        self.fix_lm2()
        self.merge()
        # self.write_output()
    def load_images(self,lm_fns):
        self.lm1 = sitk.ReadImage(str(lm_fns[0]))
        self.lm2 = sitk.ReadImage(str(lm_fns[1]))
    def fix_lm1(self):
        # self.lab1 = to_int(self.lab1)
        if hasattr(self,"remapping1"):
        # self.lab1 = sitk.Cast(self.lab1,sitk.sitkLabelUInt16)
            self.lm1 = relabel(self.lm1,self.remapping1) # merge predicted lesions in to organ so there are no holes left.j
        self.lm1 = to_int(self.lm1)
        self.lm1  = sitk.BinaryFillhole(self.lm1)

    def fix_lm2(self):
        if hasattr(self,"remapping2"):
            self.lm2 = relabel(self.lm2,self.remapping2)
        self.lm2 = to_int(self.lm2)
        # self.lm2  = sitk.BinaryFillhole(self.lm2)

    def merge(self):
        lm1_ar = sitk.GetArrayFromImage(self.lm1)
        lab2_ar = sitk.GetArrayFromImage(self.lm2)
        lm2_labels = np.unique(lab2_ar)
        lm2_labels = np.delete(lm2_labels,0)
        for label in lm2_labels:
            lm1_ar[lab2_ar==label]=label


        self.lm_merged =sitk.GetImageFromArray(lm1_ar)
        self.lm_merged = align_sitk_imgs(self.lm_merged,self.lm2)
    def write_output(self):
        print("Writing file {}".format(self.output_fname))
        sitk.WriteImage(self.lm_merged,str(self.output_fname))


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
    def __init__(self, lm,lm_fn=None,dom_label="larger",ignore_labels=[],threshold=0.1) -> None:

        '''
        threshold: 0.1 if the dominant label has less than thresholded volume contribution, it will be ignored and next largest will be selected instead .
        Whenever more than one labels are presented inside a single cc, the dom_label is assigned to all
        '''
    #HACK:  note these are oriented bounding boxes. Make sure their overlap comes out right
        super().__init__(lm,ignore_labels)
        self.fil.ComputeOrientedBoundingBoxOn()
        self.fil.Execute(self.lm_cc)
        self.lm_bin_cc, self.nbr_binary= get_1lbl_nbrhoods(self.lm_binary, 1)
        self.fil_bin = sitk.LabelShapeStatisticsImageFilter()
        self.fil_bin.ComputeOrientedBoundingBoxOn()
        self.fil_bin.Execute(self.lm_bin_cc)
        store_attr()


    def process(self):
        self.init_overlap_matrix()
        self.separate_touching_gps()
        if not self.is_empty() and self.touching_labels==True :
            self.ccs_sharing_binaryblobs()
            self.compute_dsc_touching_labels()
            self.compute_dsc_touching_labels_multiple_candidates()
            assert all(self.ccs_touching.contrib>0), "Some labels are matched to the wrong binary blob and have 0 contrib their matched group which should not possible. Filename: {}".format(self.lm_fn)
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
        self.ccs_touching=self.nbrhoods[~self.nbrhoods.cent.isin(self.nbr_binary.cent)]
        self.ccs_touching = self.ccs_touching.assign(contrib= np.nan)
        self.ccs_no_change=self.nbrhoods[self.nbrhoods.cent.isin(self.nbr_binary.cent)]
        if len(self.ccs_touching)==0: self.touching_labels=False
        else: self.touching_labels=True


        
    def ccs_sharing_binaryblobs(self):
        '''
        creates: label_pairs tuples of [cc_label,bin_label]
        '''
        
        bins  = self.nbr_binary[~self.nbr_binary.cent.isin(self.nbrhoods.cent)]
        self.label_pairs=[]
        self.label_pairs_multiple_candidates=[]
        for ind in range(len(self.ccs_touching)):
            cc1 = self.ccs_touching.iloc[ind]
            lab_cc = int(cc1.label_cc)
            if lab_cc== 0:
                print(self.lm_fn)
            lab_bin_candidates=self.find_superset_binary_label(lab_cc,bins.label_cc)
            if len(lab_bin_candidates)==1:
                cc_bin_pair = lab_cc,lab_bin_candidates[0]
                self.label_pairs.append(cc_bin_pair)
            else:
                cc_bin_pair = lab_cc,lab_bin_candidates
                self.label_pairs_multiple_candidates.append(cc_bin_pair)



    def find_superset_binary_label(self,cc_lab,bin_labs):
        bb1= self.GetBoundingBox(cc_lab)
        supersets=[]
        for bin_lab in bin_labs:
            bin_lab = int(bin_lab)
            bb2= self.fil_bin.GetBoundingBox(bin_lab)
            is_inside= bb1_inside_bb2(bb1,bb2)
            if is_inside==True: supersets.append(bin_lab)
        return supersets

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
            self.ccs_touching.loc[self.ccs_touching.label_cc==label_cc, 'contrib']= contrib


    def compute_dsc_touching_labels_multiple_candidates(self):
            if len(self.label_pairs_multiple_candidates)==0: return
            pairs_final,contribs=[],[]
            for pair in self.label_pairs_multiple_candidates:
                pairs=[]
                for cand in pair[1]:
                    argsi = [pair[0],cand]
                    pairs.append(argsi)

                args = [[self.lm_cc, self.lm_bin_cc, *a, False] for a in pairs]
                dsc_multi = multiprocess_multiarg(
                            labels_overlap, args, 16, False, False, progress_bar=True
                        )
                best_idx = np.argmax(dsc_multi)
                pair_best = pairs[best_idx]
                contribs.append(dsc_multi[best_idx])
                pairs_final.append(pair_best)

            for ind, pair in enumerate(pairs_final):
                contrib = contribs[ind]
                label_cc = pair[0]
                self.ccs_touching.loc[self.ccs_touching.label_cc==label_cc, 'contrib']= contrib


    def find_dom_labels(self):
        cc_groups = self.ccs_touching.cc_group.unique()
        for cc_group in cc_groups:
            ccs_mini = self.ccs_touching[self.ccs_touching.cc_group==cc_group]
            ccs_mini = ccs_mini.loc[ccs_mini['contrib']>self.threshold]
            dom_label = ccs_mini.label.max()
            self.ccs_touching.loc[self.ccs_touching.cc_group==cc_group,'label_out' ]= dom_label
        self.ccs_no_change = self.ccs_no_change.assign(label_out=self.ccs_no_change.label)
        self.nbrhoods_with_remapping  = pd.concat([self.ccs_touching,self.ccs_no_change])

    def create_remappings(self):
        remapping={}
        for i,row in self.nbrhoods_with_remapping.iterrows():
            rem= {int(row['label_cc']):int(row['label_out']) }
            remapping.update(rem)
        return remapping
        


class MergeTouchingLabelsFiles():
        def __init__(self, ignore_labels=[]) -> None:
            store_attr()
             
        def process_batch(self,lm_fns,output_folder=None, overwrite=False):
            lm_fns,lm_fns_out = self.filter_files(lm_fns,output_folder,overwrite)
            lm_fns_final ,lms = self.load_images(lm_fns)
            lms_fixed = []
            for i, lm in enumerate(lms) :
                    lm_fixed = self.process_lm(lm,lm_fns_final[i])
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
            lm_fns_final=[]
            for lm_fn in lm_fns:
                print("Processing {}".format(lm_fn))
                lm = sitk.ReadImage(str(lm_fn))
                lms.append(lm)
                lm_fns_final.append(lm_fn)
            return lm_fns_final, lms
        def process_lm(self,lm, lm_fn):
                    Merger = MergeTouchingLabels(lm,ignore_labels=self.ignore_labels,lm_fn=lm_fn)
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
            lm_fns, lm_fns_out = self.set_output_names(lm_fns,overwrite)
            # if overwrite==False:
            #     for fn, fn_out in zip(lm_fns, lm_fns_out):
            #         if fn_out.exists() :
            #                 print("File {} already exists. Skipping".format(fn))
            #                 lm_fns.remove(fn)
            #                 lm_fns_out.remove(fn_out)
            return lm_fns,lm_fns_out


        def set_output_names(self,lm_fns,overwrite=False):
            lm_fns_final , lm_fns_out=[], []
            for fn in lm_fns:
                lm_fn_out = self.set_output_name(fn)
                if not lm_fn_out.exists() or overwrite==True:
                    lm_fns_out.append(lm_fn_out)
                    lm_fns_final.append(fn)
                else:
                   print("File {} already exists. Skipping".format(fn))
            return lm_fns_final,lm_fns_out


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


def merger_wrapper(fns,ignore_labels,overwrite):
    M = MergeTouchingLabelsFiles(ignore_labels)
    M.process_batch(lm_fns=fns,overwrite=overwrite)

def merge_multiprocessor(lm_fns,ignore_labels =[],overwrite=False,n_chunks=12):
            # if overwrite==False:
            #         inferred_output_fldr= fixed_output_folder(lm_fns[0])
            #         need_doing = []
            #         for fn in lm_fns:
            #             fn_name = fn.name
            #             inferred_output_fname = inferred_output_fldr/fn_name
            #             if not inferred_output_fname.exists():
            #                 need_doing.append(fn)
            #         lm_fns = list(il.compress(lm_fns, need_doing))
            #
            argsi = list(chunks(lm_fns,n_chunks))
            argsi = [[a,ignore_labels, overwrite] for a in argsi]
            return multiprocess_multiarg(merger_wrapper,argsi)



           #
# fname1 =  Path('/s/datasets_bkp/litqsmall/sitk/masks/litqsmall_00000.nrrd')
# fname2 = Path( '/s/fran_storage/predictions/lits/ensemble_LITS-408_LITS-385_LITS-383_LITS-357_LITS-413/litqsmall_00000.nrrd')
# merge_multiprocessor(fname1,fname2,output_fldr,overwrite=True)
# %%
# %%
if __name__ == "__main__":
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    preds_fldr_lidc2 = Path("/s/fran_storage/predictions/lidc2/LITS-913")
    preds_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-1018")
    nnunet_fldr = Path("/s/datasets_bkp/ark/")

# %%

    fixed_fldr = Path("/s/fran_storage/predictions/litsmc/LITS-1018_fixed_mc")
    fixed_fns = list(fixed_fldr.glob("*"))
    lm_fns = list(preds_fldr.glob("crc*"))
    pending = [find_matching_fn(fn,fixed_fns) is None for fn in lm_fns]
    lm_fns_pending = list(il.compress(lm_fns,pending))
# %%

# %%
#SECTION:-------------------- Merge separate liver and lesion labelmaps (MP)--------------------------------------------------------------------------------------

    overwrite=False
    merge_multiprocessor(lm_fns= lm_fns_pending, overwrite=overwrite,n_chunks= 4, ignore_labels =[1])
# %%
# %%
#SECTION:-------------------- Fix touching labels (e.g., cyst and cancer in single lesion)--------------------------------------------------------------------------------------

    Merger = MergeTouchingLabelsFiles(ignore_labels=[1])
    Merger.process_batch(lm_fns=lm_fns,overwrite=overwrite)
# %%
#SECTION:-------------------- MORE--------------------------------------------------------------------------------------

    Mergr = MergeTouchingLabels()
    lm_fixed = Merger.process()
    sitk.WriteImage(lm_fixed,lm_fn.str_replace(".nii","_fixed.nii"))

# %%
    Merger.compute_dsc_touching_labels()
    Merger.find_dom_labels()
    remapping = Merger.create_remappings()
    Merger.lm_out = relabel(Merger.lm_cc,remapping)

# 
# %%
    L = LabelMapGeometry(Merger.lm_binary)
    L.ComputeOrientedBoundingBoxOn()
    L.Execute(L.lm_cc)

    Merger.fil.ComputeOrientedBoundingBoxOn()
    Merger.Execute(Merger.lm_cc)
    o_origin = Merger.GetOrientedBoundingBoxOrigin(65)
    os  = Merger.GetOrientedBoundingBoxSize(65)
    o_end = [a+b for a,b in zip(o_origin,os)]




    o1 = L.GetOrientedBoundingBoxOrigin(22)
    s1 =  L.GetOrientedBoundingBoxSize(22)
    e1 = [x+y for x,y in zip(o1,s1)]

# %%
    cc_lab,bin_lab = Merger.label_pairs[11]
    bin_labs = Merger.nbr_binary.label_cc.to_list()
    cc_lab = 9
    bb1= Merger.GetBoundingBox(cc_lab)

    for bin_lab in bin_labs:
        bin_lab =4
        bin_lab = int(bin_lab)
        bb2 =  Merger.fil_bin.GetBoundingBox(bin_lab)
        bb2_start= Merger.fil_bin.GetOrientedBoundingBoxOrigin(bin_lab)
        bb2_size= Merger.fil_bin.GetOrientedBoundingBoxSize(bin_lab)
        # is_inside= bb1_inside_bb2(bb1_start,bb1_size,bb2_start,bb2_size)
        is_inside = bb1_inside_bb2(bb1,bb2)

# %%
    lm = sitk.ReadImage(lm_fn)

    labels = get_labels(Merger.lm_cc)
    remapping = {x:1 for x in labels}
    lm_bin = relabel(Merger.lm_cc,remapping)
    lm_bin_cc = to_cc(lm_bin)
    labs_bin = get_labels(lm_bin_cc)

    filt = sitk.LabelShapeStatisticsImageFilter()
    filt.Execute(lm_bin_cc)
    cents = [filt.GetCentroid(x) for x in labs_bin]
# %%
    sitk.WriteImage(Merger.lm_bin_cc,"bin_cc.nii.gz")
# %%
# %%



    sitk.WriteImage(MergeLiver.lm_merged,str(MergeLiver.output_fname))
# %%
