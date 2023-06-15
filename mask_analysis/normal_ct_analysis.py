
# %%
import SimpleITK as sitk
from fran.utils.helpers import multiprocess_multiarg
from mask_analysis.helpers import remove_organ_mask
from mask_analysis.radiomics_analysis import LesionGeometry
from pathlib import Path
import pandas as pd

def lesion_stat_wrapper(fn):
        img = sitk.ReadImage(fn)
        img = remove_organ_mask(img,tumour_always_present=False)
        L = LesionGeometry(img)
        L.dust(3)
        return fn,L.n_labels,L.lengths
    


# %%
if __name__ == "__main__":

# %%
    preds_fldr = Path(
    "/s/fran_storage/predictions/lits/ensemble_LITS-451_LITS-452_LITS-453_LITS-454_LITS-456/"
    )


    normals = [fn for fn in preds_fldr.glob("*nrrd") if "normal" in fn.name]
    args = [[fn] for fn in normals]
    output =   multiprocess_multiarg(lesion_stat_wrapper, args)
    df = pd.DataFrame(data=output,columns =['filename','number of lesions','lengths'])
    df.to_csv("~/code/mask_analysis/results/normal_cases_analysis.csv")

# %%
