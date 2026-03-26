from pathlib import Path
from typing import Union
import SimpleITK as sitk

import itk
import torch
from utilz.imageviewers import ImageMaskViewer

from label_analysis.geometry_itk import BBoxInfoFromITK, LabelMapGeometryITK
from utilz.itk_sitk import ConvertSimpleItkImageToItkImage, monai_to_itk_image, monai_to_sitk_image

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


class BBoxInfoFromPT(BBoxInfoFromITK):
    '''
    Fast read-only class for nbrhoods. Do not attempt to use other methods from parent classes, not guaranteed to work
    '''

    def __init__(
        self,
        li: str|Path|torch.Tensor,
        ignore_labels=[],
        ):
        if isinstance(li,str|Path): li = torch.load(li,weights_only=False)
        li = monai_to_itk_image(li)
        ImageType = type(li)
        caster = itk.CastImageFilter[ImageType, itk.Image[itk.US, 3]].New(Input=li)
        caster.Update()
        li = caster.GetOutput()
        super().__init__(li = li, ignore_labels=ignore_labels)


# %%
if __name__ == '__main__':

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------    
    from label_analysis.geometry_itk import label_to_labelmap
    gt_fn=Path("/r/datasets/preprocessed/test/fixed_spacing/spc_080_080_150_rsc5609df8a/lms/kits23_00002.pt")
    im_fn=Path("/r/datasets/preprocessed/test/fixed_spacing/spc_080_080_150_rsc5609df8a/images/kits23_00002.pt")
    lm = torch.load(gt_fn,weights_only=False)
    im = torch.load(im_fn,weights_only=False)
    L = LabelMapGeometryPT(gt_fn,ignore_labels=[1])
    G = BBoxInfoFromPT(lm,ignore_labels=[1])
    G.li_itk
    G.li_org = G.li_itk
# %%
    G.create_li_cc(False)
    G.calc_geom()
    G.nbrhoods = G.nbrhoods[~G.nbrhoods["label_org"].isin(G.ignore_labels)]



# %%
    label_map = itk.label_image_to_label_map_filter(G.li_itk)

    labels = [
      label_map.GetNthLabelObject(i).GetLabel()
      for i in range(label_map.GetNumberOfLabelObjects())
    ]
    print(labels)

# %%
    labels = [lab for lab in labels if lab not in G.ignore_labels]
    lab = 2
    lmap =label_to_labelmap(G.li_itk,lab, compute_feret=False)
# %%
    for cc in lmap.GetLabels():
        print(cc)
# %%
        self.key[int(cc)] = lab

    self.unique_lms.append(
        {
            "lmap": lmap,
            "label_org": lab,
            "n_islands": lmap.GetNumberOfLabelObjects(),
        }
    )
# %%
    f= itk.LabelImageToLabelMapFilter[itk.Image[itk.UC, 3], itk.LabelMap[itk.StatisticsLabelObject[itk.UL, 3]]].New(
        
    )
    f.SetInput(lm)
    f.SetInputForegroundValue(1)
    f.SetComputeFeretDiameter(True)
    f.Update()
    lm = f.GetOutput()
# %%
    meta =  lm.meta
    affine = meta['affine'].detach().cpu().to(torch.float64)
    linear_ras = affine[:3, :3]
    spacing = torch.linalg.vector_norm(linear_ras, dim=0)
    spacing = torch.where(spacing == 0, torch.ones_like(spacing), spacing)
    direction_ras = linear_ras / spacing
    ras_to_lps = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    direction_lps = ras_to_lps @ direction_ras
    origin_lps = ras_to_lps @ affine[:3, 3]

    lm_arr = lm.detach().cpu()
    li = itk.GetImageFromArray(lm_arr.permute(2, 1, 0).contiguous().numpy())

    ImageType = type(li)
    info = itk.ChangeInformationImageFilter[ImageType].New(Input=li)
    info.ChangeSpacingOn()
    info.ChangeOriginOn()
    info.ChangeDirectionOn()
    info.SetOutputSpacing(tuple(float(v) for v in spacing.tolist()))
    info.SetOutputOrigin(tuple(float(v) for v in origin_lps.tolist()))
    info.SetOutputDirection(itk.matrix_from_array(direction_lps.numpy()))
    info.Update()
    li = info.GetOutput()

    L2 = LabelMapGeometryITK(li=li,ignore_labels=[1])
    bbox = L2.nbrhoods['bbox'][0]
# %%

    
    L.dust(1)
    L.nbrhoods
    L.li_cc_sitk
    L.labels
    get_labels(L.li_cc_sitk)

    L.dust(1)
# %%
    bb = (slice(bbox[0],bbox[0]+bbox[3]),slice(bbox[1],bbox[1]+bbox[4]),slice(bbox[2],bbox[2]+bbox[5]))
    lm2 = lm[bb]
    im2 = im[bb]
    ImageMaskViewer([im2,lm2])
# %%
