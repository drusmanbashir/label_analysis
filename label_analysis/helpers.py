
# %%
import SimpleITK as sitk
from fastcore.basics import listify
import ipdb
import numpy as np

from fran.utils.imageviewers import view_sitk

tr = ipdb.set_trace


def astype(id: int, inds):  # assumes arg 0 is an image
    ids = listify(id)
    inds = listify(inds)

    def wrapper(func):
        def _inner(*args, **kwargs):
            args = list(args)
            for id, ind in zip(ids, inds):
                img = args[ind]
                if any([img.GetPixelID() == 8, img.GetPixelID() == 9]) and id == 22:  # float -> label
                    img = sitk.Cast(img, sitk.sitkUInt8)
                if img.GetPixelID() != id:
                    img = sitk.Cast(img, id)
                    args[ind] = img
            return func(*args, **kwargs)

        return _inner

    return wrapper


#dangerous as missing values will not be accounted for
@astype(1, 0)
def get_labels(img):
    arr = sitk.GetArrayFromImage(img)
    arr = np.unique(arr)
    arr_int = [int(a) for a in arr if a != 0]
    return arr_int


@astype(22, 0)
def remove_organ_label(img,tumour_always_present=True):
    '''
    tumour_always_present: Set to false if dataset can have organ-only masks with no lesions present
    '''
    

    n_labs = len(get_labels(img))
    if n_labs == 2:
        img = sitk.ChangeLabelLabelMap(img, {1: 0, 2: 1})
    elif n_labs == 1 :
        if tumour_always_present==True:
            print(
                "only one label is present in this mask. Assuming that is tumour (and not organ). Nothing is changed."
            )
        else:
            img = sitk.ChangeLabelLabelMap(img, {1: 0})
    elif n_labs==0:

        print("No labels in file ")

    else:
        print("Too many labels: {}. Which one is organ?".format(n_labs))
    return img


@astype(22, 0)
def remove_labels(lm, labels):
        labels = listify(labels)
        org_type = lm.GetPixelID()
        dici = {x: 0 for x in labels}
        lm_cc = sitk.ChangeLabelLabelMap(to_label(lm), dici)
        if lm_cc.GetPixelID() != org_type:
            lm_cc = sitk.Cast(lm_cc, org_type)
        return lm_cc



def relabel(lm,remapping):
        org_type = lm.GetPixelID()
        lm_cc= to_label(lm)
        lm_cc = sitk.ChangeLabelLabelMap(lm_cc,remapping)
        try:
            if lm_cc.GetPixelID() != org_type:
                lm_cc = sitk.Cast(lm_cc, org_type)
        except:
            print("Could not recast to original pixel type {0}. Returned img is of type {1}".format(org_type, lm_cc.GetPixelID()))
        return lm_cc




@astype(22, 0)
def single_label(mask, target_label):
    labs_all = get_labels(mask)
    if len(labs_all) > 1:
        non_active = [l for l in labs_all if l != target_label]
        dici = {j: 0 for j in non_active}
    else:
        dici = {}
    active = {target_label: 1}
    dici.update(active)
    single_label = sitk.ChangeLabelLabelMap(mask, dici)
    return single_label

#
# def single_label2(mask, target_label):
#     mask_np = sitk.GetArrayFromImage(mask)
#     mask_np[mask_np != target_label] = 0
#     mask_np[mask_np == target_label] = 1
#     mask = sitk.GetImageFromArray(mask_np)
#     return mask


@astype(1, 0)
def to_cc(lm):
        lm_cc = sitk.ConnectedComponent(lm)
        return lm_cc



@astype(22, 0)
def to_label(x):
    return x


@astype(1, 0)
def to_int(x):
    return x


# %%
# %%
