from pathlib import Path
from typing import Union
import SimpleITK as sitk

import itk
import torch

from label_analysis.geometry_itk import LabelMapGeometryITK
from utilz.itk_sitk import ConvertSimpleItkImageToItkImage, monai_to_sitk_image

from label_analysis.helpers import get_labels


def pt_to_itk_image(li, pixel_type=itk.UC):
    sitk_img, src = monai_to_sitk_image(li)
    itk_img = ConvertSimpleItkImageToItkImage(sitk_img, pixel_type)
    return itk_img, src


class LabelMapGeometryPT(LabelMapGeometryITK):
    def __init__(

        self,
        li: Union[itk.Image, sitk.Image, str, Path],
        ignore_labels=[],
        img=None,
        compute_feret=True,
    ):
        itk_img, src = pt_to_itk_image(li)
        super().__init__(
            li=itk_img,
            ignore_labels=ignore_labels,
            img=img,
            compute_feret=compute_feret,
        )
        self.li_fn = src


# %%
if __name__ == '__main__':
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------    
    data_folder = Path(
    "/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/"
    )
    pred_fldr = Path("/s/fran_storage/predictions/kits/KITS-bl")
    img_fldr = data_folder / "images"
    lms_fldr = data_folder / "lms"
    imgs = sorted(img_fldr.glob("*.pt"))
    lms=sorted(lms_fldr.glob("*.pt"))
    preds = sorted(pred_fldr.glob("*.pt"))
    pred_fn = preds[0]
    pred= torch.load(pred_fn,weights_only=False)
    gt_fn=Path("/r/datasets/preprocessed/kits/lbd/spc_080_080_150_rlb00ec4022_rlb00ec4022_ex020/lms/kits23_00209.pt")# pred_fn=/s/fran_storage/predictions/kits/KITS-n7/kits23_00114.pt
    L = LabelMapGeometryPT(gt_fn,ignore_labels=[1])
# %%
    
    L.dust(1)
    L.nbrhoods
    L.li_cc_sitk
    L.labels
    get_labels(L.li_cc_sitk)

    L.dust(1)
# %%
