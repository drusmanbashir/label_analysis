# %%
from dataclasses import dataclass
from pprint import pp
import pandas as pd
from fran.data.dataregistry import DS
from label_analysis.helpers import get_labels


def remove_zero_from_list(func):
    def _inner(*args, **kwargs):
        output = func(*args, **kwargs)
        output = [x for x in output if x != 0]
        return output

    return _inner


#
#
@dataclass
class Structure:
    name: str
    label_full: int
    label_region: int | list
    label_minimal: int | list
    name_region: str

    def __getattribute__(self, structure: str):
        value = object.__getattribute__(self, structure)
        if isinstance(value, list):
            value = [v for v in value if v != 0]
        return value


#
class TotalSegmenterLabels:
    """
    Class for managing and processing labels from the TotalSegmenter dataset.
    This class provides functionality for retrieving and remapping labels
    stored in an Excel file, specificlabel_fully intended for medical imaging tasks.
    df columns: structure,, label,  label_short, name_region, name_region, side
      structure is vanilla names.
       is same but without side so both adrenals get the same label.
      name_region is general idea as in chest, abdo, label_full body (erector spinae)
      name_region/label_region: gives name_region descriptions which may be used in creawting a low-res localiser image. e.g., label_full vessels are same label
    """

    def __init__(self) -> None:
        """
        Initializes TotalSegmenterLabels by loading label data from a default Excel file.
        Loads label information into dataframes
        """
        totalseg_folder = DS["totalseg_meta"].folder
        meta_fn = totalseg_folder / "meta_codex_v2.xlsx"
        self.df = pd.read_excel(meta_fn, sheet_name="labels")
        self.meta = pd.read_excel(meta_fn, sheet_name="meta")

    @remove_zero_from_list
    def get_labels(self, organ="label_full", side=None, localiser=False):
        """
        Retrieve labels filtered by organ and side.

        Parameters:
        organ (str): The organ to filter by. Use "label_full" to get all labels.

        side (str): If specified, only labels for this side of the organ are retrieved.
        localiser (bool): If True, localiser labels are returned instead of full.

        Returns:
        List[str]: A list of label identifiers.

        Raises:
        ValueError: If specified parameters don't match any entries.
        """
        if organ == "label_full":
            return self.df.label.to_list()
        if side:
            labs = self.df.loc[
                (self.df[""] == organ) & (self.df["side"] == side)
            ]
        else:
            labs = self.df.loc[(self.df[""] == organ)]
        if localiser == False:
            labs_out = labs["label_full"].to_list()
        else:
            labs_out = labs["label_region"].to_list()
        return labs_out

    @remove_zero_from_list
    def get_label_by_name(self, name: str, group: str):
        """
        Return unique numeric `label_region` values for rows whose
        `` matches the provided word.

        Parameters
        ----------
        word : str
            Search token for ``.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("word must be a non-empty string")

        structures = self.df[""].fillna("").astype(str)
        token = name.strip()
        mask = structures.str.lower() == token.lower()

        labels = (
            self.df.loc[mask, "label_region"].dropna().astype(int).unique().tolist()
        )
        return labels

    def create_remapping(
        self,
        src_labels: str | list,
        dest_labels: str | list,
        as_dict=False,
        as_list=False,
    ) -> dict | list:

        assert as_list or as_dict, "Either list mode or dict mode should be true"
        label_cols = ["label_full", "label_region", "label_minimal"]
        if isinstance(src_labels, list) and isinstance(dest_labels, list):
            pass
        elif (
            isinstance(src_labels, str)
            and isinstance(dest_labels, str)
            and src_labels in label_cols
            and dest_labels in label_cols
        ):
            src_labels = getattr(self, src_labels)
            dest_labels = getattr(self, dest_labels)
            # pairs = set(zip(src_labels,dest_labels))
        elif (
            isinstance(src_labels, str)
            and isinstance(dest_labels, str)
            and src_labels in label_cols
        ):
            dest_struc = dest_labels
            STU = getattr(self, dest_struc)
            maps_from = getattr(STU, src_labels)
            neo = self.df[src_labels].where(self.df[src_labels].isin(maps_from), 0)
            dest_labels = neo.tolist()
            src_labels = self.df[src_labels].tolist()
            # pairs = set(zip(src_labels,dest_struc))
        else:
            raise NotImplementedError

        pairs = set(zip(src_labels, dest_labels))
        if as_dict:
            # remapping = {s: d for s, d in zip(src_labels, dest_labels)}
            remapping = {s: d for s, d in pairs}
            return remapping
        elif as_list:
            srcs = [a[0] for a in pairs]
            dests = [a[1] for a in pairs]
            return [srcs, dests]

    @property
    @remove_zero_from_list
    def label_full(self):
        """
        List[int]: Retrieve label_full label IDs.

        Returns a complete list of label_full available label identifiers in the dataset.

        """
        return self.df.label_full.to_list()

    # @property
    # def label_short(self):
    #     """
    #     List[str]: Retrieve shortened label names.
    #
    #     Returns a list of short descriptive names for each label in the dataset.
    #     """
    #     return self.df.label_short.to_list()
    @property
    @remove_zero_from_list
    def label_region(self):
        return self.df.label_region.to_list()

    @property
    @remove_zero_from_list
    def label_minimal(self):
        return self.df.label_minimal.to_list()

    def _make_structure(self, structure: str, rows: pd.DataFrame) -> Structure:
        label_loc = rows.label_region.unique().tolist()
        label = rows.label_full.unique().tolist()
        label_min = rows.label_minimal.unique().tolist()
        name_region = rows.name_region.unique().tolist()
        SN = Structure(
            name=structure,
            label_full=label,
            label_region=label_loc,
            label_minimal=label_min,
            name_region=name_region,
        )
        return SN

    @property
    def all(self) -> Structure:
        return self._make_structure("all", self.df)

    def __getattr__(self, structure: str):
        """
        Dynamic access:
        - `self.<structure_word>` -> unique localiser labels for matching 
        """

        df = self.df
        lookup_cols = ["name_full", "structure", "name_region", "name_minimal"]
        for str_cols in [col for col in lookup_cols if col in df.columns]:
            rows = df.loc[df[str_cols] == structure]
            if len(rows) > 0:
                return self._make_structure(structure, rows)

        print(f"Structure {structure} not found")
        raise AttributeError(
            f"TSL attribs can be either df column names or entries of  or structure. Full list of structure shorts: {df['name_full'].unique().tolist()}\n Full list of structures: {df['structure'].unique().tolist()}"
        )
        # labels = self.get_label_by_name(name, group)
        # if labels:
        #     return labels
        #
        #


# %%

if __name__ == "__main__":
    import re
    from pathlib import Path

    import SimpleITK as sitk
    from fran.managers.project import Project

    P = Project("totalseg")
    TSL = TotalSegmenterLabels()
    TSL.pancreas
    rem = TSL.create_remapping("label_full", "label_minimal", as_dict=True)
    rem = TSL.create_remapping("label_full", "abdomen", as_dict=True)
    TSL.create_remapping("label_region", "bladder_prostate", as_dict=True)
    TSL.create_remapping("label_minimal", "pancreas", as_dict=True)

    TSL.lung
    TSL.misc
    pp(TSL.abdomen)
# %
    fn1 = "/s/fran_storage/predictions/totalseg/LITS-827/lidc2_0001.nii.gz"
    fn2 = "/s/xnat_shadow/lidc2/masks/lidc2_0001.nii.gz"
    lm1 = sitk.ReadImage(fn1)
    lm2 = sitk.ReadImage(fn2)
    l2 = get_labels(lm2)

# %%
    mask = TSL.df[""].where("lungs", 0)
# %%
    df = TSL.df
    structure = 'gu'
    for str_cols in ["name_full","structure","name_region","name_minimal"]:
        rows = df.loc[df[str_cols] == structure]
# %%

# SECTION:-------------------- CREATING simple labelset-------------------------------------------------------------------------------------- <CR>
# %%

    df = TSL.df

    df.structure
    cr = df.structure
    pat = "_left|_right"
    ents = []
# %%
    TSL.create_remapping(TSL.label_full, TSL.lung, as_dict=True)

    # %r
    src_labels = TSL.label_region
    src_labels = ""
    dest_struc = "lung"
    dest_struc = getattr(TSL, dest_struc)
    getattr(dest_struc, src_labels)
    dest_struc = dest_struc.label_region

    neo = TSL.df["label_region"].where(TSL.df["label_region"].isin(dest_struc), 0)
    src_labels = getattr(TSL, src_labels)
    src_labels = src_labels.tolist()
# %%
    for structure in cr:
        # aa=    cr.iloc[0]
        b = re.sub(pat, "", structure)
        ents.append(b)

    df = df.assign(short=ents)
    df_n = Path("/s/datasets_bkp/totalseg/meta_new.csv")
    df.to_csv(df_n)
# %%
    as_dict = True
    src_labels = "label_region"
    dest_struc = "lung"

    assert as_list or as_dict, "Either list mode or dict mode should be true"
    if src_labels in ["label_full", "label_region"] and dest_struc in [
        "all",
        "label_region",
    ]:
        src_labels = getattr(TSL, src_labels)
        dest_struc = getattr(TSL, dest_struc)
    elif src_labels == "label_region":
        dest_struc = getattr(TSL, dest_struc)
        neo = TSL.df["label_region"].where(
            TSL.df["label_region"].isin(dest_struc), 0
        )
        dest_struc = neo.tolist()
        src_labels = TSL.df["label_region"]
    # if src_labels == dest_labels:
    #     return None
    if as_dict == True:
        remapping = {s: d for s, d in zip(src_labels, dest_struc)}
    elif as_list == True:
        rem = [src_labels, dest_struc]

# %%
# SECTION:-------------------- CREATING label_short------------------------------------------------------------------------------------- <CR>
    key = fk_generator(1)
    df = pd.read_csv(df_n)
    short = df.name_full
# %%
    dones = {}
    for structure in short:
        if not structure in dones.keys():
            label = next(key)
            dici = {structure: label}
            dones.update(dici)
# %%
    labs = []
    for structure in short:
        lab = dones[structure]
        labs.append(lab)
# %%
    df = df.assign(label_short=labs)
    df.to_csv(df_n, index=False)
# %%
    lr = TSL.get_labels("lung", "right")
    ll = TSL.get_labels("lung", "left")

    remapping = {l: 0 for l in TSL.label_full}
    print(lr)
    print(ll)
# %%
    for l in lr:
        remapping[l] = 6

    for l in ll:
        remapping[l] = 7
# %%
    lm1 = relabel(lm1, remapping)
    l3 = merge(lm1, lm2)
# %%
    structure = "lung"
    df = TSL.df
    df[df[""] == structure]
    df[df["structure"] == structure]

    rows = df.loc[df[""] == structure]
    label_loc = rows.label_region.unique().tolist()
    label = rows.label.unique().tolist()
    label_min = rows.label_minimal.unique().tolist()
    name_region = rows.name_region.unique().tolist()

    SN = Structure(
        name=structure,
        label=label,
        label_region=label_loc,
        label_minimal=label_min,
        name_region=name_region,
    )
# %%

    dest_struc = "pancreas"
    src_labels = "label_minimal"
    STU = getattr(TSL, dest_struc)
    maps_from = getattr(STU, src_labels)
    neo = TSL.df[src_labels].where(TSL.df[src_labels].isin(maps_from), 0)
    dest_struc = neo.tolist()
    src_labels = TSL.df[src_labels]
    src_labels = src_labels.tolist()
    pairs = set(zip(src_labels, dest_struc))

# %%
    def nozero(func):
        def _inner(*args, **kwargs):
            output = func(*args, **kwargs)
            print(output)
            output = output[output != 0]
            print(output)
            return output

        return _inner

# %%
    @nozero
    def listi():
        return [0, 1, 2, 3]

    listi()
# %%

# %%
