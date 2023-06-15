from mask_analysis.helpers import astype, single_label, to_int
import itertools as il
import numpy as np
import SimpleITK as sitk


@astype([5, 5], [0, 1])
def labels_overlap(mask_cc, pred_cc, lab_mask, lab_pred):
    mask2 = single_label(mask_cc, lab_mask)
    pred2 = single_label(pred_cc, lab_pred)
    fil = sitk.LabelOverlapMeasuresImageFilter()
    a, b = map(to_int, [mask2, pred2])
    fil.Execute(a, b)
    dsc, jac = fil.GetDiceCoefficient(), fil.GetJaccardCoefficient()
    indices = lab_mask - 1, lab_pred - 1
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
