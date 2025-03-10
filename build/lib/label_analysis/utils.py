# %%
from typing import Union
from pathlib import Path
from fastcore.all import is_close, listify
from fastcore.meta import test_eq
import torch
import shutil
from torch import nn

import itertools as il
from fastcore.test import test_close
import ast
from fastcore.basics import store_attr
import numpy as np
from torch.functional import Tensor
from fran.inference.helpers import get_sitk_target_size_from_spacings
from utilz.imageviewers import ImageMaskViewer
from utilz.fileio import maybe_makedirs, str_to_path

import SimpleITK as sitk
from utilz.helpers import abs_list
from utilz.string import get_extension
import ipdb

from utilz.string import cleanup_fname
tr = ipdb.set_trace
from fastcore.transform import Transform, ItemTransform
import itertools


# %%



def long_short_axes(lm_cc,label:int):
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    arr= sitk.GetArrayFromImage(lm_cc)
    pc1_x, pc1_y,pc1_z, pc2_x, pc2_y,pc2_z, pc3_x,pc3_y,pc3_z = filter_label.GetPrincipalAxes(label)

# get the center of mass
    com_y, com_x ,com_z= filter_label.GetCentroid(1)

# now trace the distance from the centroid to the edge along the principal axes
# we use some linear algebra

# get the position of each point in the image
    v_x, v_y ,v_z= np.where(arr)

# convert these positions to a vector from the centroid
    v_pts = np.array((v_x - com_x, v_y - com_y,v_z-com_z)).T

# project along the first principal component
    distances_pc1 = np.dot(v_pts, np.array((pc1_x, pc1_y,pc1_z)))

# get the extent
    dmax_1 = distances_pc1.max()
    dmin_1 = distances_pc1.min()

# project along the second principal component
    distances_pc2 = np.dot(v_pts, np.array((pc2_x, pc2_y,pc2_z)))

# get the extent
    dmax_2 = distances_pc2.max()
    dmin_2 = distances_pc2.min()

    ax1 = dmax_1-dmin_1
    ax2 = dmax_2-dmin_2
    return ax1,ax2


def distance_tuples(cent1,cent2):
        vec = np.array([a-b for a,b in zip(cent1,cent2)])
        distance = np.linalg.norm(vec )
        return distance
        

def array_to_sitk(arr:Union[Tensor,np.ndarray]):
    '''
    converts cuda to cpu. Rest is as sitk.GetImageFromArray
    '''
    if isinstance(arr,Tensor) and arr.device.type=='cuda':
        arr = arr.detach().cpu()
    return sitk.GetImageFromArray(arr)
    

class SITKDICOMOrient(Transform):    
    '''
    Re-orients SITK Images to DICOM. Allows all other datatypes to pass
    '''

    def __init__(self):
        self.dicom_orientation =  (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    def encodes(self,x:sitk.Image):
            if isinstance(x,sitk.Image):
                if x.GetDirection!=self.dicom_orientation:
                    x = sitk.DICOMOrient(x,"LPS")
            return x

class SITKImageMaskFixer():
    @str_to_path([1,2])
    def __init__(self,img_fn, mask_fn): 
        store_attr()
        self.img,self.mask = map(sitk.ReadImage,[img_fn,mask_fn])
        self.dicom_orientation =  (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    def process(self,fix=True,outname=None):
        self.essential_sitk_props()
        self.verify_img_mask_match()
        if self.match_string!="Match" and fix==True: 
            self.mask = align_sitk_imgs(self.img,self.mask)
            self.match_string="Repaired"
            self.to_DICOM_orientation()
        self.save_altered_sitk(outname)


    def save_altered_sitk(self,outname=None):
        if self.match_string =="Repaired":
            if outname:
                sitk.WriteImage(self.mask, self.mask_fn.parent/(f"{outname}_mask.nii"))
            else:
                sitk.WriteImage(self.mask,self.mask_fn)
        elif "changed to DICOM" in self.match_string:
            if not outname:
                itertools.starmap(sitk.WriteImage,[(self.img,self.img_fn),(self.mask,self.mask_fn)])
            else:
                img_fn = self.img_fn.parent/(f"{outname}_img.nii")
                mask_fn= self.img_fn.parent/(f"{outname}_mask.nii")
                itertools.starmap(sitk.WriteImage,[(self.img,img_fn),(self.mask,mask_fn)])
            

    def to_DICOM_orientation(self):
        direction = np.array(abs_list(ast.literal_eval(self.pairs[0][0])))
        try:
            test_eq(direction,self.dicom_orientation)
        except: 
            print(f"Changing img/mask orientation from {direction} to {np.eye(3)}")
            self.img, self.mask = map(
                lambda x: sitk.DICOMOrient(x, "LPS"), [self.img, self.mask]
            )
            self.match+=", changed to DICOM"
            self.pairs.insert(1,str(self.dicom_orientation)*2)

    def verify_img_mask_match(self ):
            matches =[] 
            for l in self.pairs:
                l = [ast.literal_eval(la) for la in l]
                match = is_close(*l,eps=1e-5)
                matches.append(match)

            if all(matches):         self.match_string = 'Match' 
            elif not matches[0]: raise Exception("Irreconciable difference in sizes. Check img/mask pair {}".format(self.mask_fn))
            else:
                self.match_string= 'Mismatch'
           
    def essential_sitk_props(self):
            directions,sizes,spacings = [],[],[]
            for arr in self.img,self.mask:
                directions.append(str(abs_list(arr.GetDirection())))
                sizes.append(str(arr.GetSize()))
                _spacing = arr.GetSpacing()
                _spacing = list(map(lambda x: round(x,2), _spacing))
                spacings.append(str(_spacing))
            self.pairs =[directions] + [sizes] + [spacings]

    @property 
    def log(self):
        return [self.match_string]+[self.img_fn,self.mask_fn]+self.pairs

def set_sitk_props(img:sitk.Image,sitk_props:Union[list,tuple])->sitk.Image:
        origin,spacing,direction = sitk_props
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img
def align_sitk_imgs(img,img_template):
                    img = set_sitk_props(img,[img_template.GetOrigin(),img_template.GetSpacing(),img_template.GetDirection()])
                    # img.CopyInformation(img_template)
                    return img


def create_sitk_as(img:sitk.Image,arr:Union[np.array,Tensor]=None)->sitk.Image:
    if arr is not None:
        img_new = sitk.GetImageFromArray(arr)
    else:
        img_new = sitk.Image(*img.GetSize())
    img_new = align_sitk_imgs(img_new,img)
    return img_new

def get_metadata(img:sitk.Image)->list   :
    res = img.GetSize(), img.GetOrigin(), img.GetSpacing(), img.GetDirection()
    return  res

def fix_slicer_labelmap(mask_fn,img_fn):
    '''
    slicer output labelmaps are not full sized but just a bbox of the labels
    this function zero-fills outside the bbox to match imge size
    
    '''
    print("Processing {}".format(mask_fn))
    img = sitk.ReadImage(img_fn)
    mask = sitk.ReadImage(mask_fn)
    assert(all([a==b for a,b in zip(img.GetSpacing(),mask.GetSpacing())])), "Different source files \nImage:{0}--->Spacings{1} \nMask: {2}--->Spacings{3}".format(img_fn.name,img.GetSpacing(),mask_fn.name,mask.GetSpacing())
    m = sitk.GetArrayFromImage(mask)
    i_shape = sitk.GetArrayFromImage(img).shape
    m_shape = m.shape
    if i_shape==m_shape:
        print("Identical shaped image and mask. Nothing done")
    else:
        print("Image shape is {0}. Mask shape is {1}. Creating mask backup in /tmp folder and fixing..".format(i_shape,m_shape))
        mask_bk_fn = Path("/tmp")/mask_fn.name
        shutil.copy(mask_fn,mask_bk_fn)
        distance =[a-b for a,b in zip(mask.GetOrigin(),img.GetOrigin())]
        ad = [d/s for d,s in zip(distance,img.GetSpacing())]
        ad.reverse()
        ad_int = [int(a) for a  in ad]
        test_close(ad,ad_int)
        shp = list(img.GetSize())
        shp.reverse()
        zers = np.zeros(shp)
        zers[ad_int[0]:ad_int[0]+m_shape[0],ad_int[1]:ad_int[1]+m_shape[1],ad_int[2]:ad_int[2]+m_shape[2]] = m
        mask_neo = create_sitk_as(img,zers)
        sitk.WriteImage(mask_neo,mask_fn)

@str_to_path(0)
def compress_img(img_fn):
    '''
    if img_fn is a symlink. This will alter the target file
    '''
    img_fn = img_fn.resolve()
    
    e = get_extension(img_fn)
    fn_neo = img_fn.str_replace(e,"nii.gz")
    fn_old_bkp = img_fn.str_replace("/s","/s/tmp")
    fn = Path("/s/xnat_shadow/litq/images/litq_48_20200107.nii.gz")
    
    for parent in fn_old_bkp.parents:
        maybe_makedirs(parent)
    if e!="nii.gz":
        print("Converting {0}  -----> {1}".format(img_fn,fn_neo))
        img = sitk.ReadImage(img_fn)
        sitk.WriteImage(img,fn_neo)
        shutil.move(img_fn,fn_old_bkp)

        
    else:
        print("File {} already nii.gz format. Nothing to do.".format(img_fn))
    

def cast_folder(fldr,pixel_id=1):
    '''
    params pixel_id: sitk.sutkUInt8 is 1
    
    '''
    imgs = list(fldr.glob("*"))
    for fn in imgs:
        img = sitk.ReadImage(fn)
        if not img.GetPixelID() == pixel_id:
            print(img.GetPixelIDTypeAsString())
            img = sitk.Cast(img, sitk.sitkUInt8)
            sitk.WriteImage(img, fn)


@str_to_path(0)
def compress_fldr(fldr:Path, recursive=True):
    if recursive==True:
        files = fldr.rglob("*")
    else:
        files = fldr.glob("*")
    for fn in files:
        if fn.is_file():
            compress_img(fn)


def thicken_nii(niifn,max_thickness=3.0):
        print("Processing file {}".format(niifn))
        im_ni = sitk.ReadImage(niifn)

        st_org = im_ni.GetSpacing()[-1]
        
        if st_org>= max_thickness:
            print("Already thick slice-image ({0}mm). Skipping {1}".format(st_org,niifn))
        else:
            if st_org >0.9 and st_org<1.5:
                step=3
            elif st_org <0.9:
                step=5

            elif st_org >=1.5 and st_org <max_thickness:
                step =2
            im_np= sitk.GetArrayFromImage(im_ni)
            im = torch.tensor(im_np).float()
            n_slice = im_np.shape[0]
            in_plane = im_np.shape[1:]   
            im1d = im.view(n_slice,-1)
            im1d= im1d.unsqueeze(1)
            av = nn.Conv1d(1,1, 3,padding=1)
            filt = nn.parameter.Parameter(torch.ones_like(av.weight))
            av.weight = filt
            imthic = av(im1d)
            imthic = imthic.view(-1,*in_plane)
            imthic = imthic[::step,:]
            imthic_np = imthic.detach().numpy()
            im_out = sitk.GetImageFromArray(imthic_np)
            im_out = align_sitk_imgs(im_out,im_ni)
            outthickness  = st_org*step
            outthickness = np.minimum(max_thickness,outthickness)
            spacing= im_out.GetSpacing()
            spacing = (spacing[0],spacing[1],outthickness)
            im_out.SetSpacing(spacing)
            print("Starting nslices: {0}. Final nslices: {1}".format(n_slice,imthic_np.shape[0]))
            return im_out

def convert_to_gz(infldr,outfldr=None):
    if not outfldr: outfldr = infldr
    for fn in fldr.glob("*nii"):
        print("Processing file {}".format(fn))
        fn_out_name = fn.name.replace("nii","nii.gz")
        fn_out = outfldr/(fn_out_name)
        img = sitk.ReadImage(fn)
        sitk.WriteImage(img,fn_out)


def has_target_labels(mask_fn,labels=[1,2]):
    '''
    based on the dataset checks if label 0,1,... are present in  the given file
    some files are missing label 0 (bg) or 1  (fg). This checks that so you can fix those files
    '''
    print("Processing {}".format(mask_fn))
    labels=listify(labels)
    
    mask = sitk.ReadImage(mask_fn)
    mask_np = sitk.GetArrayFromImage(mask)
    absent = [lab not in mask_np for lab in labels]
    f = list(il.compress(labels,absent))
    if len(f)>0:
        print("Labels abset: {}".format(f))
        return 0
    else:
        print("All labels are present")
        return 1
  
class SITKImgMaskResize(ItemTransform):
    def __init__(self,out_spacing):
        self.resizer = SITKResize(out_spacing)
    def encodes(self,x):
        img,mask = x
        img_out = self.resizer(img,is_label=False)
        mask_out = self.resizer(mask,is_label=True)
        return img_out,mask_out

class SITKResize(Transform):
    def __init__(self,out_spacing):
        store_attr()
    def encodes(self,img_sitk,is_label):
        out_size = get_sitk_target_size_from_spacings(img_sitk,self.out_spacing)
        _ ,out_origin,  _ ,out_direction= get_metadata(img_sitk)
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(out_direction)
        resample.SetOutputOrigin(out_origin)
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(img_sitk.GetPixelIDValue())
        if is_label==True:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        img_out= resample.Execute(img_sitk)
        return img_out




# %%
if __name__ == "__main__":
# %%
    fldr = Path("/s/xnat_shadow/litq/masks")
    files = list(fldr.glob("*"))
    mask_fn = files[1]
# %%
    for fn in files:
        has_target_labels(fn,[1,2])

# %%

    img = sitk.ReadImage(mask_fn)
    img.GetPixelIDTypeAsString()
    fil = sitk.LabelShapeStatisticsImageFilter()
    fil.ComputeFeretDiameterOn()
    fil.Execute(img)
    labels = fil.GetLabels()
# %%

# %%


# %%
    f_in="/s/insync/datasets/crc_project/masks_ub/crc_CRC003_20181026_CAP1p5_thick.nii.gz"
    ref_img = sitk.ReadImage(fn)
    labelmap_thick =sitk.ReadImage(f_in)
# %%
    img_fn= "/s/fran_storage/datasets/raw_data/lax/images/lits_5.nii"
    ref_img = sitk.ReadImage(img_fn)
# %%
    is_label=False
    out_size,out_origin, spacing,out_direction= get_metadata(ref_img)
    o_size,o_origin, o_spacing,o_direction= get_metadata(img_out)
    out_spacing = (0.8,0.8,1.5)
    out_size = get_sitk_target_size_from_spacings(ref_img,out_spacing)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(out_direction)
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(ref_img.GetPixelIDValue())

# %%
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    img_out= resample.Execute(ref_img)

    mask_fn_out="/s/insync/datasets/crc_project/masks_ub/crc_CRC003_20181026_CAP1p5_thin.nii.gz"
    sitk.WriteImage(label_thin,mask_fn_out

                    )

    tt = ToTensorT()
    aa = tt.encodes(img_out)
# %%
    masks_fldr = Path("/home/ub/Desktop/capestart/nodes/Capestart_Nodes-Output/")
    images_fldr = Path("/home/ub/Desktop/capestart/nodes/batch2/images")
    masks = list(masks_fldr.glob("*"))
    imgs = list(images_fldr.glob("*"))
    processed=0
    for mask_fn in masks:
        mt = cleanup_fname(mask_fn.name)
        img_fn =[fn for fn in imgs if cleanup_fname(fn.name)==mt]
        if len(img_fn)>1 :
            tr()
        elif len(img_fn)==1:
            fix_slicer_labelmap(mask_fn,img_fn[0])
            processed+=1
        else:
            print("No matching image for {}".format(mask_fn))
    print(processed)
# %%
     


    fldr = Path("/s/datasets_bkp/drli/masks/")
    for fn in fldr.glob("*"):
        imgfn = fn.str_replace("masks","images")
        if not imgfn.exists():
            tr()
# %%

    img_fn =  Path('/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii')
    mask_fn = Path("/media/ub/UB11/datasets/lits_short/segmentation-51.nii")
    fldr = "/s/datasets_bkp/crc_project/nifti/masks_ub/finalised/"
    compress_fldr(fldr)

# %%

    img_fn = Path("/s/fran_storage/datasets/raw_data/lits2/images/litq_77_20210306.nii.gz")
    mask_fn = "/s/fran_storage/datasets/raw_data/lits2/masks/litq_77_20210306.nrrd"
    mask_outfn = "/s/fran_storage/datasets/raw_data/lits2/masks/litq_77_20210306_fixed.nrrd"
    img = sitk.ReadImage(img_fn)
    np_a = sitk.GetArrayFromImage(img)
    np_a = np_a.transpose(2,1,0)
    
    np2 = np.mean(np_a,1)
    np2 = np2.transpose(0,1)
    plt.imshow(np2)
    ImageMaskViewer([a,a])

