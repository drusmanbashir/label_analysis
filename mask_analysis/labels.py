# %%
from pandas import read_json
from fran.utils.fileio import load_json
from fran.utils.helpers import pp
from mask_analysis.helpers import astype, get_labels, single_label, to_cc, to_int
import itertools as il
import numpy as np
import SimpleITK as sitk
import ipdb
tr = ipdb.set_trace



@astype([5, 5], [0, 1])
def labels_overlap(gt_cc, pred_cc, lab_gt, lab_pred):
    gt_all_labels = get_labels(gt_cc)
    assert lab_gt in gt_all_labels, "Label {} is not present in the Groundtruth ".format(lab_gt)
    mask2 = single_label(gt_cc, lab_gt)
    pred2 = single_label(pred_cc, lab_pred)
    fil = sitk.LabelOverlapMeasuresImageFilter()
    a, b = map(to_int, [mask2, pred2])
    fil.Execute(a, b)
    dsc, jac = fil.GetDiceCoefficient(), fil.GetJaccardCoefficient()
    indices = lab_gt - 1, lab_pred - 1
    return dsc, jac, indices


#
# def compute_overlap_perlesion(labsA,labsB):
#         print("Computing label jaccard and dice scores")
#         # get jaccard and dice
#         lab_inds = list(il.product(labsA, labsB))
#         dsc = np.zeros((len(labsB), len(labsA))).transpose()
#         jac = np.copy(self.dsc)
#         args = [[self.mask_cc_dusted, self.pred_cc_dusted, *a] for a in lab_inds]
#         d = multiprocess_multiarg(
#             labels_overlap(), args, 16, False, False, progress_bar=True
#         )  # multiprocess i s slow
#
#         for sc in d:
#             ind_pair = sc[2]
#             self.dsc[ind_pair] = sc[0]
#             self.jac[ind_pair] = sc[1]
#

# %%
if __name__ == "__main__":
    pass

    
# %%
