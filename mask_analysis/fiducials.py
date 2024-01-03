# %%
from pathlib import Path
from fastcore.all import store_true
from fastcore.basics import store_attr
from mask_analysis.mergemasks import MergeMasks
from pandas import unique
from pywt import data
from pyxnat.core.resources import shutil
from fran.utils.fileio import load_json, maybe_makedirs
from fran.utils.helpers import find_matching_fn, pp
from fran.utils.imageviewers import view_sitk
from mask_analysis.helpers import astype, get_labels, remove_organ_label, single_label, to_cc, to_int
import itertools as il
import numpy as np
import SimpleITK as sitk
import ipdb
import pandas as pd
from fran.utils.sitk_utils import align_sitk_imgs

from fran.utils.string import info_from_filename, int_to_str, strip_extension
tr = ipdb.set_trace


def identical_key_vals(dicti):
    return list(dicti.keys()) == list(dicti.values())

class RemapFromDF():
    def __init__(self,df: pd.DataFrame,dataset_fldr: Path, target_label:int = 2, schema = None ,fixed_labels= [0,1]):
        store_attr()
        self.mapping = {label:label for label in fixed_labels}
        if self.schema is None:
            self.schema =  {'benign': 2, 'mets': 3}
        self.out_fldr = dataset_fldr / "masks_multiclass"
        self.mask_fldr = dataset_fldr / "masks"
        self.img_fldr = dataset_fldr / "images"
        self.img_fns  = list(self.img_fldr.glob("*"))
        self.mask_fns = list(self.mask_fldr.glob("*"))
        self.markup_fns = list((dataset_fldr / "markups").glob("*"))

        additional_columns=[]
        for row in self.df.itertuples():
            additional_columns.append(self.get_matching_fns(row))

        self.df = pd.concat([self.df, pd.DataFrame(additional_columns)], axis=1)

        maybe_makedirs(self.out_fldr)

    def process(self):
        for ind in range(len(self.df)):
            row = self.df.iloc[ind]
            if row.lesion_labels=='na':
                pass
            else:
                self.process_row(row)

    def process_row(self,row, overwrite=False):
        case_id = row.case_id
        print("Remapping {} to {}".format(row.mask_fn, row.mask_fn_out))
        if not overwrite and Path(row.mask_fn_out).exists():
            print("File {} exists. Skipping..".format(row.mask_fn_out))
            return
        if row.lesion_labels == 'json':
            R = Remap(organ_label=1)
            R.process(row.mask_fn, row.mask_fn_out, row.markup_fn)
        else:
            case_mapping = self.mapping|{self.target_label: self.schema[row.lesion_labels]}
            if identical_key_vals(case_mapping):
                print("Identcal mapping for {0}' by default. No remapping needed. Making copy in {1}".format(case_id, self.out_fldr))
                shutil.copy(row.mask_fn, row.mask_fn_out)
            else:
                print("Remapping schema: {}".format(case_mapping))
                mask = sitk.ReadImage(row.mask_fn)
                mask = sitk.Cast(mask, sitk.sitkUInt8)
                mask = sitk.ChangeLabel(mask,case_mapping)
                sitk.WriteImage(mask, row.mask_fn_out)
                print("Done")

    def get_matching_fns(self,row):
            assert row.lesion_labels in ['benign', 'mets', 'na', 'json'], "Illegal lesion label: {}".format(row.lesion_labels)
            if row.lesion_labels=='na':
                return {'img_fn':None, 'mask_fn':None, 'markup_fn':None, 'mask_fn_out':None}
            else:
                case_id = row.case_id
                img_fn = [fn for fn in self.img_fns if info_from_filename(fn.name)['case_id'] == case_id ]
                assert len(img_fn)==1, "multiple (or none) images found for {}".format(case_id)
                img_fn = img_fn[0]
                mask_fn = find_matching_fn(img_fn,self.mask_fns)
                if row.lesion_labels=='json':
                    markup_fn = [fn for fn in self.markup_fns if info_from_filename(fn.name)['case_id'] == case_id ]
                    assert len(markup_fn)==1, "multiple (or None) markups found for {}".format(case_id)
                    markup_fn = markup_fn[0]
                else:
                    markup_fn = None

                if case_id == '00033': tr()
                mask_fn_out= self.out_fldr/(mask_fn.name)
                dici = {'img_fn':img_fn, 'mask_fn':mask_fn, 'markup_fn':markup_fn, 'mask_fn_out':mask_fn_out}
                return dici



class Remap():
    def __init__(self, organ_label:int=None):
        store_attr()
        self.fil = sitk.LabelShapeStatisticsImageFilter()
        self.fil.SetComputeFeretDiameter(True)


    def preprocess_lm(self,lm):
        self.lm_bkp  = lm
        lm_cc = sitk.Image(self.lm_bkp)
        lm_cc = to_int(lm_cc)
        if self.organ_label is not None:
            remove_mapping = {self.organ_label:0}
            lm_cc = sitk.ChangeLabel(lm_cc,remove_mapping) 
        lm_cc = to_cc(lm_cc)
        return lm_cc

    def get_fid_locs(self,slicer_markups):
        mu = slicer_markups['markups']
        fid_info = mu[0]['controlPoints']
        fid_locs = [a['position'] for a in fid_info]
        fid_locs = np.array(fid_locs)
        return fid_locs

    def process(self,lm_fn,lm_fn_out , markup_fn,overwrite=False):
            if lm_fn_out.exists() and not overwrite:
                print("File {} exists. Skipping..".format(lm_fn_out))
                return
            markups = load_json(markup_fn)
            markup_type ='mets' if 'mets' in markup_fn.name else 'benign'
            if markup_type=='mets' :
                fid_label, non_fid_label = 3,2
            else:
                fid_label, non_fid_label = 2,3

            lm = sitk.ReadImage(lm_fn)
            lm_cc = self.relabel(lm, markups, fid_label, non_fid_label)
            print("Writing {}".format(lm_fn_out))
            sitk.WriteImage(lm_cc, lm_fn_out)



    def relabel(self, lm, fid_markups, fid_label, non_fid_label):
    
        lm_cc = self.preprocess_lm(lm)
        self.fil.Execute(lm_cc)
        labels = self.fil.GetLabels()
        centroids = [self.fil.GetCentroid(lab) for lab in labels]
        centroids = np.array(centroids)

        radii= [self.fil.GetFeretDiameter(lab)/2 for lab in labels]

        fid_locs = self.get_fid_locs(fid_markups)
        distance_vecs = np.array([a-centroids for a in fid_locs])
        distances = np.linalg.norm(distance_vecs, axis=2)
        closest_markers = ( distances < radii)
        closest_marker_indices = np.nonzero(closest_markers)[1]+1
        remapping = {x:non_fid_label for x in labels}
        for key in closest_marker_indices:
            remapping[key] = fid_label 

        #assert((len(closest_markers)) == (len(lu:=np.unique(closest_markers)))), "Repeat values Some fiducials are closest to more than one lesion. It may be an orphan or from other category"
        lm_cc = sitk.ChangeLabel(lm_cc,remapping)
        if self.organ_label is not None:
            lm_cc = self.put_organ_back(lm_cc)
        return lm_cc

    def put_organ_back(self,lm_cc):
        organ = sitk.GetArrayFromImage(self.lm_bkp)
        lesions_cc = sitk.GetArrayFromImage(lm_cc)

        lesions_cc[organ==1]=1
        lm_cc = sitk.GetImageFromArray(lesions_cc)
        lm_cc = align_sitk_imgs(lm_cc,self.lm_bkp)
        del self.lm_bkp
        return lm_cc

# %%
if __name__ == "__main__":

# %% [markdown]
## AIscreening dataset
# %%

    # df_fn = Path("/s/datasets_bkp/lits_segs_improved/segs_notes.csv")
    imgs_fldr = Path("/s/datasets_bkp/litqsmall/images")
    df_fn = Path("/s/datasets_bkp/litqsmall/litqsmall_notes.csv")
    df = pd.read_csv(df_fn)
    aa = df.case_id
    bb= aa.str.split("_",expand=True)
    df.case_id = bb[1]
    df.dropna(subset= ['case_id','lesion_labels'],inplace=True)
    # int_to_str(ĳkĳk)

    print(df)
# %%

    RD = RemapFromDF(df,imgs_fldr.parent)

    print(RD.df['case_id','lesion_labels','img_fn'])
    RD.process()
# %%
# %%

    info_from_filename(Path("/s/datasets_bkp/litqsmall/images/litqsmall_00036.nii.gz").name)
# %%
# %%

    view_sitk(lm,lm_cc)
#
