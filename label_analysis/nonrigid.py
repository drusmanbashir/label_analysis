# %%
import itk
import vtk
import SimpleITK as sitk
import numpy as np

from vtkmodules.vtkImagingCore import vtkImageWrapPad
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import (
    VTK_VERSION_NUMBER,
    vtkVersion
)
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkFlyingEdges3D,
    vtkMarchingCubes,
    vtkThreshold
)

import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkDataSetAttributes
)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkImagingCore import vtkImageWrapPad
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import vtkDICOMImageReader, vtkMetaImageReader, vtkNIFTIImageReader
from vtkmodules.vtkImagingHybrid import vtkVoxelModeller
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

from fran.utils.imageviewers import view_sitk

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
import numpy as np

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


# %%
if __name__ == '__main__':
    fish_target = np.loadtxt('/home/ub/Downloads/pycpd-master/data/fish_target.txt')
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:, :-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:, :-1] = fish_target
    X = np.vstack((X1, X2))

    fish_source = np.loadtxt('/home/ub/Downloads/pycpd-master/data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:, :-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:, :-1] = fish_source
    Y = np.vstack((Y1, Y2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()


# %%
def read_image(fn):
    reader = vtkNIFTIImageReader()
    reader.SetFileName(fn)
    reader.Update()
    im = reader.GetOutput()
    return im

def march(image,threshold=1):
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()
    imo = mc.GetOutput()
    return imo

def write_polydata(poly,fn):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(fn)
    writer.Update()

from pathlib import Path
from fran.utils.string import strip_extension
# %%
if __name__ == "__main__":

    fn1_im = "/s/xnat_shadow/crc/images/crc_CRC003_20181026_CAP1p5.nii.gz"
    fn1_lm = "/s/fran_storage/predictions/litsmc/LITS-933_fixed_mc/crc_CRC003_20181026_CAP1p5.nii.gz"
    fn2_lm = (
        "/s/fran_storage/predictions/litsmc/LITS-933/crc_CRC162_20150425_ABDOMEN.nii.gz"
    )
    fn2_im = "/s/xnat_shadow/crc/images/crc_CRC162_20150425_ABDOMEN.nii.gz"

    fn3_im = "/s/xnat_shadow/crc/images/crc_CRC275_20161229_CAP1p51.nii.gz"
    fn3_lm = (
        "/s/fran_storage/predictions/litsmc/LITS-933/crc_CRC275_20161229_CAP1p51.nii.gz"
    )
# %%
    reader = vtkNIFTIImageReader()
    reader.SetFileName(fn1_lm)
    reader.Update()
    reader.GetQFormMatrix()

    im1 = read_image(fn1_lm)

# %%
    im2 = read_image(fn2_lm)
    c2 = march(im2)
    fn2_out= strip_extension(Path(fn2_lm).name)+".stl"
    fn1_out= strip_extension(Path(fn1_lm).name)+".stl"
    write_polydata(c2,fn2_out)

# %%
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(c1)
    icp.SetTarget(c2)
    icp.SetMaximumNumberOfIterations(50)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()

    icp.Update()

# %%

    mat = icp.GetMatrix()
    translation=[icp.GetMatrix().GetElement(0,3),icp.GetMatrix().GetElement(1,3),icp.GetMatrix().GetElement(2,3)]
    w_translation = np.zeros([1,3])
    spacing=im1.GetSpacing()
    origin=im1.GetOrigin()
    w_translation = np.array(origin)+np.array(spacing)*np.array(translation)
    outTx_itk=sitk.Euler3DTransform()
    outTx_itk.SetTranslation(w_translation)

# %%
    im1 = sitk.ReadImage(fn1_lm)
    im2 =  sitk.ReadImage(fn2_lm) 
    im1_t= sitk.Resample(im1, im2, outTx_itk, sitk.sitkNearestNeighbor, 0.0, im1.GetPixelID())
# %%
    view_sitk(im1_t,im1_t,data_types=['mask','mask'])

# %%
    tm = icp.GetMatrix()
    icpTransformFilter = vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(c1)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    c1_t= icpTransformFilter.GetOutput()
    out_fn = strip_extension(Path(fn1_lm).name)+"_t.stl"
    write_polydata(c1_t,out_fn)

    colors = vtkNamedColors()
    extent = im.GetExtent()
    pad = vtkImageWrapPad()

    pad.SetInputConnection(reader.GetOutputPort())
    pad.SetOutputWholeExtent(extent[0], extent[1] + 1, extent[2], extent[3] + 1, extent[4], extent[5] + 1)
    pad.Update()

    # Copy the scalar point data of the volume into the scalar cell data
    pad.GetOutput().GetCellData().SetScalars(reader.GetOutput().GetPointData().GetScalars())
    im.GetPointData()
    cell = pad.GetOutput().GetCellData()

    icp.SetSource(im)
# %%
    start_label = 1
    end_label = 1
    selector = vtkThreshold()
    selector.SetInputArrayToProcess(0, 0, 0, vtkDataObject().FIELD_ASSOCIATION_CELLS,

                                    vtkDataSetAttributes().SCALARS)
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
    render_window.SetWindowName('GenerateCubesFromLabels')

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d('DarkSlateBlue'))
    render_window.Render()

    camera = renderer.GetActiveCamera()
    camera.SetPosition(42.301174, 939.893457, -124.005030)
    camera.SetFocalPoint(224.697134, 221.301653, 146.823706)
    camera.SetViewUp(0.262286, -0.281321, -0.923073)
    camera.SetDistance(789.297581)
    camera.SetClippingRange(168.744328, 1509.660206)

    render_window_interactor.Start()

    surface = vtkMarchingCubes()
#
