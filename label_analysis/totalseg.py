from label_analysis.helpers import get_labels, relabel
from label_analysis.merge import merge
import pandas as pd
from pathlib import Path
import SimpleITK as sitk





class TotalSegmenterLabels():
    def __init__(self) -> None:
        meta_fn = Path("/s/datasets_bkp/totalseg/meta.xlsx")
        self.df= pd.read_excel(meta_fn, sheet_name="labels")
        self.meta =  pd.read_excel(meta_fn, sheet_name="meta")

    def labels(self,organ,side=None):
        if side:
            labs = self.df.loc[(self.df['organ']==organ) & (self.df['side']==side)]
        else:
            labs = self.df.loc[(self.df['organ']==organ) ]
        labs_out = labs['label'].to_list()
        return labs_out

    def create_remapping(self,labelsets,labels_out):
        '''
        only labels in the list are kept all others are mapped to zero
        
        '''
        assert len(labelsets)==len(labels_out), "Make sure the labelsets and labels_out have the same length"
        remapping  = {l:0 for l in self.all}
        for lset,lout in zip(labelsets,labels_out):
            for l in lset:
                remapping[l]=lout
        return remapping



    @property
    def all(self):return list(range(1,118))



if __name__ == "__main__":
    from fran.managers.project import Project
    P =Project ('tmp')
    TSL = TotalSegmenterLabels()

    fn1 = "/s/fran_storage/predictions/totalseg/LITS-827/lidc2_0001.nii.gz"
    fn2 = "/s/xnat_shadow/lidc2/masks/lidc2_0001.nii.gz"
    lm1 = sitk.ReadImage(fn1)
    lm2 = sitk.ReadImage(fn2)
    l2 = get_labels(lm2)

# %%
# %%
    lr= TSL.labels("lung","right")
    ll = TSL.labels("lung","left")

    remapping  = {l:0 for l in TSL.all}
    print(lr)
    print(ll)
# %%
    for l in lr:
        remapping[l]= 6

    for l in ll:
        remapping[l]= 7
# %%
    lm1 = relabel(lm1,remapping)
    l3 = merge(lm1,lm2)
# %%

 
