import SimpleITK as sitk

from label_analysis.helpers import get_labels
from label_analysis.utils import align_sitk_imgs


def fill_holes_multiclass(fn, out_fn=None):
    if not out_fn:
        out_fn = fn

    lm = sitk.ReadImage(fn)
    lm_arr = sitk.GetArrayFromImage(lm)
    labs = get_labels(lm)
    lm_org = sitk.Image(lm)

    remapping = {l: 1 for l in labs if l != 1}
    lm_org = sitk.ChangeLabel(lm_org, remapping)
    lm_org = sitk.BinaryFillhole(lm_org)
    org_arr = sitk.GetArrayFromImage(lm_org)

    for lab in labs:
        org_arr[lm_arr == lab] = lab

    lm2 = sitk.GetImageFromArray(org_arr)
    lm2 = align_sitk_imgs(lm2, lm)
    print("Writing {}".format(out_fn))
    sitk.WriteImage(lm2, out_fn)
