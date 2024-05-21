# %%
import itk
from vtk.util.numpy_support import vtk_to_numpy as vnp
import numpy as np
import SimpleITK as sitk
import vtk
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from label_analysis.helpers import arrayFromVTKMatrix, crop_center
from label_analysis.utils import align_sitk_imgs
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import VTK_VERSION_NUMBER, vtkVersion
from vtkmodules.vtkCommonDataModel import (vtkDataObject, vtkDataSetAttributes,
                                           vtkImageData)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (vtkFlyingEdges3D, vtkMarchingCubes,
                                       vtkThreshold)
from vtkmodules.vtkFiltersGeneral import (vtkTransformFilter,
                                          vtkTransformPolyDataFilter)
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkImagingCore import vtkImageWrapPad
from vtkmodules.vtkImagingHybrid import vtkVoxelModeller
from vtkmodules.vtkIOImage import (vtkDICOMImageReader, vtkMetaImageReader,
                                   vtkNIFTIImageReader)
from vtkmodules.vtkRenderingCore import (vtkActor, vtkPolyDataMapper,
                                         vtkRenderer, vtkRenderWindow,
                                         vtkRenderWindowInteractor)

from fran.transforms.spatialtransforms import crop_to_bbox
from fran.utils.helpers import pp
from fran.utils.image_utils import get_bbox_from_mask
from fran.utils.imageviewers import view_sitk


# %%
def read_image(fn):
    reader = vtkNIFTIImageReader()
    reader.SetFileName(fn)
    reader.Update()
    im = reader.GetOutput()
    return im


def march(image, threshold=1):
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()
    imo = mc.GetOutput()
    return imo


def write_polydata(poly, fn):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()


from pathlib import Path

from fran.utils.string import strip_extension

# %%
if __name__ == "__main__":

    fn1_im = Path("/home/ub/code/label_analysis/testfiles/images/crc_CRC003_20181026_CAP1p5.nii.gz")

    fn2_im = Path("/home/ub/code/label_analysis/testfiles/images/crc_CRC275_20161229_CAP1p51.nii.gz")
    fn1_lm = Path("/home/ub/code/label_analysis/testfiles/lms/crc_CRC003_20181026_CAP1p5.nii.gz")
    fn2_lm = Path(
        "/home/ub/code/label_analysis/testfiles/lms/crc_CRC275_20161229_CAP1p51.nii.gz"
    )
# %%
    def center_origin(fn):
        im = sitk.ReadImage(fn)
        im.SetOrigin([0,0,0])
        sz = im.GetSize()
        im.SetOrigin([-x/2 for x in sz])
        sitk.WriteImage(im,fn)

    def zero_origin(fn):
        im = sitk.ReadImage(fn)
        im.SetOrigin([0,0,0])
        sitk.WriteImage(im,fn)

# %%
    zero_origin(fn2_lm)
    # lm1.SetOrigin([0, 0, 0])
    # lm2.SetOrigin([0, 0, 0])
    # sitk.WriteImage(lm1,fn1_im)
    # sitk.WriteImage(lm2,fn2_im)
# %%
    reader = vtkNIFTIImageReader()
    reader.SetFileName(fn1_lm)
    reader.Update()
    reader.GetQFormMatrix()

    im1 = read_image(fn1_lm)
    c1 = march(im1)
    write_polydata(c1, strip_extension(Path(fn1_lm).name) + ".stl")
# %%
    im2 = read_image(fn2_lm)
    c2 = march(im2)
    fn2_out = strip_extension(Path(fn2_lm).name) + ".stl"
    fn1_out = strip_extension(Path(fn1_lm).name) + ".stl"
    write_polydata(c2, fn2_out)

# %%
    im1 = sitk.ReadImage(fn1_lm)
    im2 = sitk.ReadImage(fn2_lm)
# %%
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(c1)
    icp.SetTarget(c2)
    icp.SetMaximumNumberOfIterations(500)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()

    icp.Update()

# %%
    spacing = np.ones(3)
    spacing = im2.GetSpacing()
    mat_ = icp.GetMatrix()
# %%
    arr =     arrayFromVTKMatrix(mat_)
    mat = arr[:3,:3]
    mat = (mat.transpose()/spacing).transpose()
# %%

    trans = arr[:3,3] 
    trans = trans/spacing
    trans = np.zeros(3)
    trans = np.expand_dims(trans,0)
    
    params  = np.concatenate([mat,trans],0)
    params2 = params.reshape(-1)

# %%
    tx = sitk.AffineTransform(3)
    tx.SetParameters(params2)

# %%
    im1_t = sitk.Resample(
        im1, im2, tx, sitk.sitkNearestNeighbor, 0.0, im1.GetPixelID()
    )
# %%
    view_sitk(im2, im1_t, data_types=["mask", "mask"])
    # view_sitk(im1, im1_t, data_types=["img", "img"])
# %%
    tm = icp.GetMatrix()
    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(c1)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    c1_t = icpTransformFilter.GetOutput()
    out_fn = strip_extension(Path(fn1_lm).name) + "_t.stl"
    write_polydata(c1_t, out_fn)

    colors = vtkNamedColors()
    extent = im.GetExtent()
    pad = vtkImageWrapPad()

    pad.SetInputConnection(reader.GetOutputPort())
    pad.SetOutputWholeExtent(
        extent[0], extent[1] + 1, extent[2], extent[3] + 1, extent[4], extent[5] + 1
    )
    pad.Update()

    # Copy the scalar point data of the volume into the scalar cell data
    pad.GetOutput().GetCellData().SetScalars(
        reader.GetOutput().GetPointData().GetScalars()
    )
    im.GetPointData()
    cell = pad.GetOutput().GetCellData()

    icp.SetSource(im)
# %%
    start_label = 1
    end_label = 1
    selector = vtkThreshold()
    selector.SetInputArrayToProcess(
        0, 0, 0, vtkDataObject().FIELD_ASSOCIATION_CELLS, vtkDataSetAttributes().SCALARS
    )
    selector.SetInputConnection(pad.GetOutputPort())
    selector.SetLowerThreshold(start_label)
    selector.SetUpperThreshold(end_label)
    selector.Update()

# %%
    # Shift the geometry by 1/2
    transform = vtkTransform()
    transform.Translate(-0.5, -0.5, -0.5)

    transform_model = vtkTransformFilter()
    transform_model.SetTransform(transform)
    transform_model.SetInputConnection(selector.GetOutputPort())

    geometry = vtkGeometryFilter()
    geometry.SetInputConnection(transform_model.GetOutputPort())

# %%
    mapper = vtkPolyDataMapper()
    # mapper.SetInputConnection(geometry.GetOutputPort())
    mapper.SetInputData(c2)
    mapper.SetScalarRange(start_label, end_label)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToMapScalars()
    actor = vtkActor()
    actor.SetMapper(mapper)

# %%
    renderer = vtkRenderer()
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(640, 480)
    render_window.SetWindowName("GenerateCubesFromLabels")

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("DarkSlateBlue"))
    render_window.Render()

    camera = renderer.GetActiveCamera()
    camera.SetPosition(42.301174, 939.893457, -124.005030)
    camera.SetFocalPoint(224.697134, 221.301653, 146.823706)
    camera.SetViewUp(0.262286, -0.281321, -0.923073)
    camera.SetDistance(789.297581)
    camera.SetClippingRange(168.744328, 1509.660206)

    render_window_interactor.Start()

    surface = vtkMarchingCubes()
# %%
