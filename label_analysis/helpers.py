
# %%
import SimpleITK as sitk
from fastcore.basics import listify
import ipdb
import numpy as np

from fran.utils.imageviewers import view_sitk

tr = ipdb.set_trace

def inds_to_labels(inds:list):
    inds = list(inds)
    # adds 1 to each 
    labels  = [x+1 for x in inds]
    return labels



def np_to_native(number):
    if hasattr(number,"dtype"):
        number = number.item()
    return number

def np_to_native_dict(remapping):
    remapping_out = {}
    for k,v in remapping.items():
        k = np_to_native(k)
        v = np_to_native(v)
        remapping_out[k] = v
    return remapping_out



def bb_limits(bb):
    bb_start = bb[:3]
    bb_end = [st+en for st,en in zip(bb_start,bb[3:])]
    return bb_start,bb_end


def bb1_inside_bb2(bb1,bb2):
    theta = lambda x,y,z: (z-y)/(x-y)
    bb1_start,bb1_end = bb_limits(bb1)
    bb2_start,bb2_end = bb_limits(bb2)
    thetas_bb1_start= [theta(x,y,z) for x,y,z in zip(bb2_start,bb2_end,bb1_start)]
    thetas_bb1_end= [theta(x,y,z) for x,y,z in zip(bb2_start,bb2_end,bb1_end)]
    thetas = np.array([thetas_bb1_start,thetas_bb1_end])
    is_inside = np.all((thetas >=0) & (thetas <=1))
    return is_inside



def bb1_intersects_bb2(bb1,bb2):
    
    bb1_start,bb1_end = bb_limits(bb1)
    bb2_start,bb2_end = bb_limits(bb2)
    ints =[]
    for ind in range(3):
        bb1x = np.arange(bb1_start[ind],bb1_end[ind])
        bb2x = np.arange(bb2_start[ind],bb2_end[ind])
        intersec =     len(np.intersect1d(bb1x,bb2x))>0
        ints.append(intersec)
    return all(ints)

def remap_single_label(lm, target_label, starting_ind):
    target_label = np_to_native(target_label)
    lm_tmp = single_label(lm, target_label)
    lm_tmp = to_cc(lm_tmp)
    labs = get_labels(lm_tmp)
    remapping = {l: l + starting_ind for l in labs}
    lm_tmp = relabel(lm_tmp, remapping)
    lm_tmp = to_label(lm_tmp)
    return lm_tmp, list(remapping.values())

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
            lm_cc = to_int(lm_cc)
            print("Could not recast to original pixel type {0}. Returned img is of type {1}".format(org_type, lm_cc.GetPixelID()))
        return lm_cc



@astype(22, 0)
def single_label(mask, target_label):
    target_label = np_to_native(target_label)
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
