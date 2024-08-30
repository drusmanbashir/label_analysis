
# %%
import re
from label_analysis.helpers import get_labels, relabel
from label_analysis.merge import merge
from label_analysis.overlap import fk_generator
import pandas as pd
from pathlib import Path
import SimpleITK as sitk





class TotalSegmenterLabels():
    def __init__(self) -> None:
        meta_fn = Path("/s/datasets_bkp/totalseg/meta.xlsx")
        self.df= pd.read_excel(meta_fn, sheet_name="labels")
        self.meta =  pd.read_excel(meta_fn, sheet_name="meta")

    def labels(self,organ="all",side=None):
        if organ=="all":
            return self.df.label.to_list()
        if side:
            labs = self.df.loc[(self.df['structure_short']==organ) & (self.df['side']==side)]
        else:
            labs = self.df.loc[(self.df['structure_short']==organ) ]
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
    def all(self):return self.df.label.to_list()


# %%

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
#SECTION:-------------------- CREATING simple labelset--------------------------------------------------------------------------------------
    df = TSL.df

    df.structure
    cr = df.structure
    pat = "_left|_right"
    ents=[]
# %%
    for aa in cr:
        # aa=    cr.iloc[0]
        b = re.sub(pat,"",aa)
        ents.append(b)

    df = df.assign(short = ents)
    df_n = Path("/s/datasets_bkp/totalseg/meta_new.csv")
    df.to_csv(df_n)
# %%
    key = fk_generator(1)
    df = pd.read_csv(df_n)
    short = df.short_name
    dones={}
    for aa in short:
        if not aa in dones.keys():
            label = next(key)
            dici = {aa:label}
            dones.update(dici)
# %%
    labs =[]
    for aa in short:
        lab = dones[aa]
        labs.append(lab)
    df = df.assign(label_short= labs)
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

 
