import SimpleITK as sitk
from fastcore.basics import listify
import ipdb

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


@astype(1, 0)
def get_labels(img):
    maxmin = sitk.MinimumMaximumImageFilter()
    maxmin.Execute(img)
    labs = int(maxmin.GetMaximum())
    return list(range(1, labs + 1))


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


@astype(1, 0)
def to_cc(img):
    return sitk.ConnectedComponent(img)


@astype(22, 0)
def to_label(x):
    return x


@astype(1, 0)
def to_int(x):
    return x


# %%
# %%
