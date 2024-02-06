# %%
from fran.utils.imageviewers import view_sitk
from fran.utils.fileio import load_dict
from fran.utils.helpers import get_pbar
import SimpleITK as sitk

from fran.utils.string import strip_extension

class  Jack():
    def __init__(self,jack):
        self.jack = jack

    def out(self):
        print (self.jack)



# %%
if __name__ == "__main__":


    fn = "/s/datasets_bkp/totalseg/s1221/ct.nii.gz"
    im = sitk.ReadImage(fn)
    view_sitk(im,im)


# %%
