
# %%
from pathlib import Path
import SimpleITK as sitk
from mask_analysis.utils import compress_fldr

from radiomics import collections

from fran.utils.string import strip_extension
fl1 = Path("/s/datasets_bkp/lits_segs_improved/images")
fl2 = Path("/s/datasets_bkp/lits_segs_improved/masks")


# %%
imgs = list(fl1.glob("*"))
masks = list(fl2.glob("*"))
# %%
compress_fldr(fl1)
# %%

imgs = [strip_extension(fn.name) for fn in imgs]
masks = [strip_extension(fn.name) for fn  in masks]

my_list = masks
duplicates = list(set([x for x in my_list if my_list.count(x)]))
print(set(masks).difference(set(imgs)))
print([item for item, count in collections.Counter(masks).items() if count > 1])
# %%
len(set(masks))

