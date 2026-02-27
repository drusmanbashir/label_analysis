# %%
import re
from label_analysis.helpers import get_labels, relabel
from label_analysis.merge import merge
from label_analysis.overlap import fk_generator
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from fran.data.dataregistry import DS
from dataclasses import dataclass, field
#
#
@dataclass
class Structure:
    name: str
    label: int
    label_localiser: int|list
    label_minimal: int|list
    location: str

#
#
# class TotalSegmenterLabels():
#     def __init__(self) -> None:
#         meta_fn = Path("/s/datasets_bkp/totalseg/meta.xlsx")
#         self.df= pd.read_excel(meta_fn, sheet_name="labels")
#         self.meta =  pd.read_excel(meta_fn, sheet_name="meta")
#
#     def labels(self,organ="all",side=None):
#         if organ=="all":
#             return self.df.label.to_list()
#         if side:
#             labs = self.df.loc[(self.df['structure_short']==organ) & (self.df['side']==side)]
#         else:
#             labs = self.df.loc[(self.df['structure_short']==organ) ]
#         labs_out = labs['label'].to_list()
#         return labs_out
#
#     def create_remapping(self,labelsets,labels_out):
#         '''
#         only labels in the list are kept all others are mapped to zero
#         
#         '''
#         assert len(labelsets)==len(labels_out), "Make sure the labelsets and labels_out have the same length"
#         remapping  = {l:0 for l in self.all}
#         for lset,lout in zip(labelsets,labels_out):
#             for l in lset:
#                 remapping[l]=lout
#         return remapping
#
#     @property
#     def all(self):return self.df.label.to_list()
#
#     @property
#     def labelshort(self):return self.df.label_short.to_list()
#
class TotalSegmenterLabels:
    """
    Class for managing and processing labels from the TotalSegmenter dataset.
    This class provides functionality for retrieving and remapping labels
    stored in an Excel file, specifically intended for medical imaging tasks.
    df columns: structure,structure_short, label,  label_short, location_localiser, location, side
      structure is vanilla names. 
      structure_short is same but without side so both adrenals get the same label.
      location is general idea as in chest, abdo, all body (erector spinae) 
      location_localiser/label_localiser: gives location descriptions which may be used in creawting a low-res localiser image. e.g., all vessels are same label
    """

    def __init__(self) -> None:
        """
        Initializes TotalSegmenterLabels by loading label data from a default Excel file.
        Loads label information into dataframes
        """
        totalseg_folder = DS['totalseg_meta'].folder
        meta_fn = totalseg_folder / "meta_codex.xlsx"
        self.df = pd.read_excel(meta_fn, sheet_name="labels")
        self.meta = pd.read_excel(meta_fn, sheet_name="meta")

    def get_labels(self, organ="all", side=None, localiser=False):
        """
        Retrieve labels filtered by organ and side.

        Parameters:
        organ (str): The organ to filter by. Use "all" to get all labels.

        side (str): If specified, only labels for this side of the organ are retrieved.
        localiser (bool): If True, localiser labels are returned instead of full.

        Returns:
        List[str]: A list of label identifiers.

        Raises:
        ValueError: If specified parameters don't match any entries.
        """
        if organ == "all":
            return self.df.label.to_list()
        if side:
            labs = self.df.loc[(self.df['structure_short'] == organ) & (self.df['side'] == side)]
        else:
            labs = self.df.loc[(self.df['structure_short'] == organ)]
        if localiser==False:
            labs_out = labs['label'].to_list()
        else:
            labs_out = labs['label_localiser'].to_list()
        return labs_out

    def get_label_by_name(self, name: str, group:str):
        """
        Return unique numeric `label_localiser` values for rows whose
        `structure_short` matches the provided word.

        Parameters
        ----------
        word : str
            Search token for `structure_short`.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("word must be a non-empty string")

        structures = self.df["structure_short"].fillna("").astype(str)
        token = name.strip()
        mask = structures.str.lower() == token.lower()

        labels = (
            self.df.loc[mask, "label_localiser"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        return labels

    def create_remapping(self,src_labels:str|list,dest_labels:str|list, as_dict=False,as_list=False):
        assert as_list or as_dict, "Either list mode or dict mode should be true"
        if src_labels in ["all","label_localiser"] and dest_labels in ["all","label_localiser","label_minimal"]:
            src_labels = getattr(self,src_labels)
            dest_labels = getattr(self,dest_labels)
            pairs = set(zip(src_labels,dest_labels))
        elif src_labels in ["label_localiser", "all"]:
            dest_labels = getattr(self,dest_labels)
            dest_labels = getattr(dest_labels,src_labels)
            neo = self.df[src_labels].where(self.df[src_labels].isin(dest_labels),0)
            dest_labels = neo.tolist()
            src_labels = self.df[src_labels]
            src_labels = src_labels.tolist()
            pairs = set(zip(src_labels,dest_labels))

        else:
            pairs = set(zip(src_labels,dest_labels))
        # if src_labels == dest_labels:
        #     return None
        if as_dict == True:
            # remapping = {s: d for s, d in zip(src_labels, dest_labels)}
            remapping = {s: d for s, d in pairs}
            return remapping
        elif as_list == True:
            srcs = [a[0] for a in pairs]
            dests = [a[1] for a in pairs]
            return [srcs,dests]


    @property
    def all(self):
        """
        List[int]: Retrieve all label IDs.

        Returns a complete list of all available label identifiers in the dataset.

        """
        return self.df.label.to_list()
    # @property
    # def label_short(self):
    #     """
    #     List[str]: Retrieve shortened label names.
    #
    #     Returns a list of short descriptive names for each label in the dataset.
    #     """
    #     return self.df.label_short.to_list()
    @property
    def label_localiser(self):
        return self.df.label_localiser.to_list()

    @property 
    def label_minimal(self):
        return self.df.label_minimal.to_list()

    def __getattr__(self, structure: str):
        """
        Dynamic access:
        - `TSL.<structure_word>` -> unique localiser labels for matching structure_short
        """

        df = self.df

        for str_cols in ["structure_short","structure", "location_localiser", "location"]:
            rows = df.loc[df[str_cols] == structure]
            if len(rows) > 0:
                label_loc = rows.label_localiser.unique().tolist()
                label= rows.label.unique().tolist()
                label_min = rows.label_minimal.unique().tolist()
                location  =rows.location.unique().tolist()
                SN= Structure(name= structure, label= label, label_localiser=label_loc, label_minimal=label_min, location=location)
                return SN
        raise AttributeError(f"TSL attribs can be either df column names or entries of structure_short or structure. Full list of structure shorts: {df['structure_short'].unique().tolist()}\n Full list of structures: {df['structure'].unique().tolist()}")
        # labels = self.get_label_by_name(name, group)
        # if labels:
        #     return labels
        # 
        #
# %%

if __name__ == "__main__":
    from fran.managers.project import Project
    P =Project ("totalseg")
    TSL = TotalSegmenterLabels()
    rem = TSL.create_remapping("all","label_minimal", as_dict=True)

    labs = TSL.lung

    print(TSL.gi)
    TSL.create_remapping("label_localiser", "gu", as_dict=True)

# %%
    fn1 = "/s/fran_storage/predictions/totalseg/LITS-827/lidc2_0001.nii.gz"
    fn2 = "/s/xnat_shadow/lidc2/masks/lidc2_0001.nii.gz"
    lm1 = sitk.ReadImage(fn1)
    lm2 = sitk.ReadImage(fn2)
    l2 = get_labels(lm2)

# %%
    mask = TSL.df['structure_short'].where("lungs", 0)
# %%
#SECTION:-------------------- CREATING simple labelset--------------------------------------------------------------------------------------

    df = TSL.df

    df.structure
    cr = df.structure
    pat = "_left|_right"
    ents=[]
# %%
    TSL.create_remapping(TSL.all, TSL.lung,as_dict=True


        
        
    )
# %r
    src_labels = TSL.label_localiser
    src_labels = ""
    dest_labels = "lung"
    dest_labels = getattr(TSL,dest_labels)
    getattr(dest_labels, src_labels)
    dest_labels = dest_labels.label_localiser


    neo = TSL.df["label_localiser"].where(TSL.df["label_localiser"].isin(dest_labels),0)
    src_labels = getattr(TSL,src_labels)
    src_labels = src_labels.tolist()
# %%
    for structure in cr:
        # aa=    cr.iloc[0]
        b = re.sub(pat,"",structure)
        ents.append(b)

    df = df.assign(short = ents)
    df_n = Path("/s/datasets_bkp/totalseg/meta_new.csv")
    df.to_csv(df_n)
# %%
    as_dict = True
    src_labels= "label_localiser"
    dest_labels = "lung"

    assert as_list or as_dict, "Either list mode or dict mode should be true"
    if src_labels in ["all","label_localiser"] and dest_labels in ["all","label_localiser"]:
        src_labels = getattr(TSL,src_labels)
        dest_labels = getattr(TSL,dest_labels)
    elif src_labels == "label_localiser":
        dest_labels = getattr(TSL,dest_labels)
        neo = TSL.df["label_localiser"].where(TSL.df["label_localiser"].isin(dest_labels),0)
        dest_labels = neo.tolist()
        src_labels = TSL.df["label_localiser"]
    # if src_labels == dest_labels:
    #     return None
    if as_dict == True:
        remapping = {s: d for s, d in zip(src_labels, dest_labels)}
    elif as_list == True:
        rem= [src_labels,dest_labels]

# %%
#SECTION:-------------------- CREATING label_short-------------------------------------------------------------------------------------
    key = fk_generator(1)
    df = pd.read_csv(df_n)
    short = df.structure_short
# %%
    dones={}
    for structure in short:
        if not structure in dones.keys():
            label = next(key)
            dici = {structure:label}
            dones.update(dici)
# %%
    labs =[]
    for structure in short:
        lab = dones[structure]
        labs.append(lab)
# %%
    df = df.assign(label_short= labs)
    df.to_csv(df_n,index=False)
# %%
    lr= TSL.get_labels("lung","right")
    ll = TSL.get_labels("lung","left")

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
    structure = "lung"
    df = TSL.df
    df[df["structure_short"] == structure]
    df[df["structure"] == structure]

    rows = df.loc[df["structure_short" ] == structure]
    label_loc = rows.label_localiser.unique().tolist()
    label= rows.label.unique().tolist()
    label_min = rows.label_minimal.unique().tolist()
    location  =rows.location.unique().tolist()

    SN= Structure(name= structure, label= label, label_localiser=label_loc, label_minimal=label_min, location=location)
# %%

 
