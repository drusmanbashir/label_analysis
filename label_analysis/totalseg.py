# %%
import re
from label_analysis.helpers import get_labels, relabel
from label_analysis.merge import merge
from label_analysis.overlap import fk_generator
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
#
#
#
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
        meta_fn = Path("/s/datasets_bkp/totalseg/meta.xlsx")
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

    #CODE: This takes an overcomplicated list of lists. Simplify this structure, see if you can trial it in mix datasets  (see #1)
    def create_remapping(self, labelsets, labels_out,localiser=False) -> dict:
        """
        Create a remapping dictionary to map specified labels to new values, mapping others to zero.

        Parameters:
        labelsets (List[List[int]]): A list of lists, where each sublist contains label IDs to remap.
        labels_out (List[int]): A list of new label values corresponding to each set in labelsets. 
        if len(labelsets) is 2, labels_out will be a list of size 2 regardless of size of sublists inside labelsets

        Returns:
        Dict[int, int]: A dictionary mapping from old labels to new labels.

        Raises:
        AssertionError: If labelsets and labels_out lengths do not match.
        Example:
            imported_labelsets = [TSL.labels("all")]
            remapping = TSL.create_remapping(imported_labelsets, [9,]*len(imported_labelsets))

            remapping = TSL.create_remapping([TSL.all],[TSL.label_minimal])
        """
        assert len(labelsets) == len(labels_out), "Make sure the labelsets and labels_out have the same length"
        if localiser==True:

            remapping = {l: 0 for l in self.label_localiser}
        else:
            remapping = {l: 0 for l in self.all}
        for lset, lout in zip(labelsets, labels_out):
            for l in lset:
                remapping[l] = lout
        return remapping


    def create_remapping    (self,src_labels,dest_labels, as_dict=False,as_list=False):
        assert as_list or as_dict, "Either list mode or dict mode should be true"
        src_labels = getattr(self,src_labels)
        dest_labels = getattr(self,dest_labels)
        if src_labels == dest_labels:
            return None
        if as_dict == True:
            remapping = {s: d for s, d in zip(src_labels, dest_labels)}
            return remapping
        elif as_list == True:
            return [src_labels,dest_labels]


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
    def lungs(self):
        lungs = self.df["structure_short"].isin(["lung"])
        labels = self.df.loc[lungs,'label_localiser'].unique().tolist()
        return labels

    @property
    def lung(self):
        return self.lungs

    @property 
    def label_minimal(self):
        return self.df.label_minimal.to_list()

# %%

if __name__ == "__main__":
    from fran.managers.project import Project
    P =Project ('tmp')
    TSL = TotalSegmenterLabels()

# %%
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
# %%
#SECTION:-------------------- CREATING label_short-------------------------------------------------------------------------------------
    key = fk_generator(1)
    df = pd.read_csv(df_n)
    short = df.structure_short
# %%
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

 
