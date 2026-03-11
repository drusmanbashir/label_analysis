# %%
import argparse
import ipdb
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkProperty

from label_analysis.helpers import get_labels
tr = ipdb.set_trace

import vtk
import SimpleITK as sitk
from vtk.util.numpy_support import numpy_to_vtk
try:
    from fran.utils.colour_palette import colour_palette
except Exception:
    colour_palette = {}


CT_PRESETS = {
    "outline": {"kind": "outline"},
    "skin": {
        "kind": "surface",
        "threshold": 350,
        "color": (0.94, 0.72, 0.63),
        "opacity": 0.20,
    },
    "soft_tissue": {
        "kind": "surface",
        "threshold": 100,
        "color": (0.82, 0.62, 0.56),
        "opacity": 0.12,
    },
    "bone": {
        "kind": "surface",
        "threshold": 250,
        "color": (0.95, 0.95, 0.90),
        "opacity": 0.22,
    },
    "skeleton": {
        "kind": "surface",
        "threshold": 700,
        "color": (0.98, 0.97, 0.92),
        "opacity": 0.35,
    },
}

def normalize_color(color):
    """Normalize an RGB color from [0, 255] to [0, 1]."""
    return [c / 255.0 for c in color]

def resample_to_isotropic(sitk_image, new_spacing=(0.8, 0.8, 0.8)):
    """
    Resamples a SimpleITK image to isotropic spacing using nearest-neighbor interpolation.
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest-neighbor interpolation
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    
    return resampler.Execute(sitk_image)

def create_volume_actor(ct_image, opacity=0.2):
    """
    Creates a volume actor for a CT scan, with translucency.
    """
    # Convert the SimpleITK image to VTK image
    vtk_image = sitk_to_vtk_image(ct_image)
    
    # Volume rendering properties
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)

    # Transfer function for opacity (for translucent volume rendering)
    opacity_transfer = vtk.vtkPiecewiseFunction()
    opacity_transfer.AddPoint(0, 0.0)    # Background is fully transparent
    opacity_transfer.AddPoint(500, 0.2)  # Soft tissues are translucent
    opacity_transfer.AddPoint(3000, 0.8) # High intensities are more opaque

    # Transfer function for color (grayscale)
    color_transfer = vtk.vtkColorTransferFunction()
    color_transfer.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
    color_transfer.AddRGBPoint(0, 0.5, 0.5, 0.5)
    color_transfer.AddRGBPoint(3000, 1.0, 1.0, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer)
    volume_property.SetScalarOpacity(opacity_transfer)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

def create_outline_actor(vtk_image):
    outline_data = vtkOutlineFilter()
    outline_data.SetInputData(vtk_image)

    map_outline = vtkPolyDataMapper()
    map_outline.SetInputConnection(outline_data.GetOutputPort())

    outline = vtkActor()
    outline.SetMapper(map_outline)
    outline.GetProperty().SetColor(vtkNamedColors().GetColor3d("Black"))

    return outline


def create_surface_actor_for_ct(vtk_image, threshold, color, opacity=0.2):
    """
    Creates a surface actor for a CT scan using contour filtering for surface extraction.
    """
    extractor = vtkMarchingCubes()
    extractor.SetInputData(vtk_image)
    extractor.SetValue(0, threshold)
    extractor.ComputeNormalsOn()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(extractor.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    back_prop = vtkProperty()
    back_prop.SetDiffuseColor(color)
    actor.SetBackfaceProperty(back_prop)
    return actor

def sitk_to_vtk_image(sitk_image):
    """
    Converts a SimpleITK image to a VTK image data format.
    """
    volume_data = sitk.GetArrayFromImage(sitk_image)
    vtk_data = numpy_to_vtk(volume_data.ravel(), deep=True)
    
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(volume_data.shape[::-1])
    vtk_image.GetPointData().SetScalars(vtk_data)
    
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    vtk_image.SetSpacing(spacing[2], spacing[1], spacing[0])
    vtk_image.SetOrigin(origin[2], origin[1], origin[0])
    
    return vtk_image


def create_ct_actors(ct_image, presets):
    vtk_image = sitk_to_vtk_image(ct_image)
    actors = []
    for preset_name in presets:
        preset = CT_PRESETS[preset_name]
        if preset["kind"] == "outline":
            actors.append(create_outline_actor(vtk_image))
        else:
            actors.append(
                create_surface_actor_for_ct(
                    vtk_image,
                    threshold=preset["threshold"],
                    color=preset["color"],
                    opacity=preset["opacity"],
                )
            )
    return actors

def create_actor_for_label(vtk_image, label, color, opacity=0.5):
    """
    Creates a VTK actor for a specific label in the 3D volume.
    """
    # Generate binary mask for the current label
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(vtk_image)
    threshold.ThresholdBetween(label, label)
    threshold.ReplaceInOn()
    threshold.SetInValue(1)  # Set inside label to 1
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)  # Set outside label to 0
    threshold.Update()

    # Extract surface with marching cubes
    contour = vtk.vtkMarchingCubes()
    # contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(threshold.GetOutputPort())
    contour.SetValue(0, 0.5)  # Surface for binary mask
    contour.ComputeNormalsOn()
        # Apply smoothing filter to the contour
    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputConnection(contour.GetOutputPort())
    smooth_filter.SetNumberOfIterations(20)  # Increase for smoother surface
    smooth_filter.SetRelaxationFactor(0.1)   # Controls smoothing intensity
    smooth_filter.FeatureEdgeSmoothingOn()
    smooth_filter.BoundarySmoothingOn()
    smooth_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smooth_filter.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    return actor

def render_volume(
    segmentation_image,
    ct_image=None,
    output_filename="rendered_image.jpg",
    ct_presets=None,
):
    """
    Renders multiple labels in a 3D SimpleITK volume, with unique colors for each label.
    """
    vtk_image = sitk_to_vtk_image(segmentation_image)
    background_color = vtk.vtkNamedColors().GetColor3d("Cornsilk")
    renderer = vtk.vtkRenderer()

    if ct_image is not None:
        for actor in create_ct_actors(ct_image, ct_presets or ["outline"]):
            renderer.AddActor(actor)

    renderer.SetBackground(background_color)

    # unique_labels = set(sitk.GetArrayViewFromImage(segmentation_image).flatten())
    unique_labels = get_labels(segmentation_image)
    for label in unique_labels:
        if label in colour_palette:
            rgb_color = colour_palette[label]
            color = normalize_color(rgb_color)  # Normalize color to 0-1
            # color = [.0,.0,1.0]
            actor = create_actor_for_label(vtk_image, label, color)
            renderer.AddActor(actor)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Adjust camera for better visualization
    camera = renderer.GetActiveCamera()

    camera.SetPosition(0, -1, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)
    renderer.ResetCamera()

    render_window.Render()
    
    # Save render to JPG
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    writer = vtk.vtkJPEGWriter()
    writer.SetFileName(output_filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    interactor.Start()

def main():
    parser = argparse.ArgumentParser(description="Render a 3D visualization for a labelmap.")
    parser.add_argument("labelmap", help="Path to the segmentation labelmap file.")
    parser.add_argument(
        "--ct",
        dest="ct_path",
        help="Optional path to a CT image used as rendering context.",
    )
    parser.add_argument(
        "--output",
        default="rendered_image.jpg",
        help="Output JPG filename. Default: rendered_image.jpg",
    )
    parser.add_argument(
        "--ct-preset",
        dest="ct_presets",
        action="append",
        choices=sorted(CT_PRESETS.keys()),
        help="Optional CT rendering preset. Repeat to combine presets, e.g. --ct-preset skeleton --ct-preset skin.",
    )
    args = parser.parse_args()

    labelmap = sitk.ReadImage(args.labelmap)
    ct_image = sitk.ReadImage(args.ct_path) if args.ct_path else None
    render_volume(
        labelmap,
        ct_image=ct_image,
        output_filename=args.output,
        ct_presets=args.ct_presets,
    )


# %%
if __name__ == '__main__':
    main()
    render_window.Render()
    interactor.Start()

    im_fn = "/s/xnat_shadow/nodes/images/nodes_43_20220805_CAP1p5SoftTissue.nii.gz"
    lm_fn = "/s/fran_storage/predictions/nodes/LITS-1405_LITS-1416_LITS-1417/nodes_42_20230425_CAP1p5mm.nii.gz"
    im = sitk.ReadImage(im_fn)
    lm = sitk.ReadImage(lm_fn)
    render_volume(lm,im,ct_presets=["skeleton"])


# Define opacity transfer function for the volume
    opacity_transfer = vtk.vtkPiecewiseFunction()
    opacity_transfer.AddPoint(-1000, 0.0)  # Air, fully transparent
    opacity_transfer.AddPoint(0, 0.1)      # Soft tissue, slightly visible
    opacity_transfer.AddPoint(300, 0.3)    # Denser tissue, more opaque
    opacity_transfer.AddPoint(1500, 0.8)   # Bone, mostly opaque

# Define color transfer function for grayscale volume rendering
    color_transfer = vtk.vtkColorTransferFunction()
    color_transfer.AddRGBPoint(-1000, 0.0, 0.0, 0.0)  # Black for air
    color_transfer.AddRGBPoint(0, 0.5, 0.5, 0.5)      # Gray for soft tissue
    color_transfer.AddRGBPoint(300, 0.9, 0.9, 0.9)    # Lighter gray for denser tissue
    color_transfer.AddRGBPoint(1500, 1.0, 1.0, 1.0)   # White for bone

# Set up volume properties
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer)
    volume_property.SetScalarOpacity(opacity_transfer)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

# Create the volume actor
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

# Rendering setup
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    renderer.AddVolume(volume)
    renderer.SetBackground(1, 1, 1)  # White background

# Adjust camera for a good view
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, -1, 1)
    camera.SetFocalPoint(0, 0, 0)
    renderer.ResetCamera()

# Render the volume
    render_window.Render()
    interactor.Start()
# %%
    lm = resample_to_isotropic(lm, (.8,.8,.8))
    im = resample_to_isotropic(im,(0.8,0.8,0.8))

    render_volume(lm,im,  output_filename="rendered_image.jpg")
# %%
    sphereSource = vtk.vtkSphereSource()
    stats = sitk.StatisticsImageFilter()
    stats.Execute(im)

    min_intensity = stats.GetMinimum()
    max_intensity = stats.GetMaximum()

# %%

    lm = resample_to_isotropic(lm, (.8,.8,.8))
    sitk_image = lm

    vtk_image = sitk_to_vtk_image(sitk_image)

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(vtk_image.GetNumberOfCells())
    for c in range(vtk_image.GetNumberOfCells()):
        colors.SetTuple(c, [0, 0, 255])

    # Extract unique labels in the image and create actors for each
    unique_labels = get_labels(sitk_image)
    label = 1
    color = colour_palette[label]
            # color = normalize_color(rgb_color)  # Normalize the color to 0-1 range
    threshold = vtk.vtkImageThreshold()
    cellData = vtk_image.GetCellData()
    cellData.SetScalars(colors)
    colors = vtkNamedColors()
    # Set the background color.
    bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])
    colors.SetColor("BkgColor", *bkg)
# %%
    threshold.SetInputData(vtk_image)
    threshold.ThresholdBetween(label, label)  # Isolate specific label
    threshold.ReplaceInOn()
    threshold.SetInValue(1)  # Set inside value to 1
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)  # Set outside value to 0
    threshold.Update()

    contour = vtk.vtkMarchingCubes()
    # contour.SetInputConnection(threshold.GetOutputPort())
    contour.SetInputData(vtk_image)
    contour.ComputeNormalsOn()
    # contour.SetInputConnection(vtk_image.GetOutputPort())
    contour.SetValue(0, 0.5)  # Generate contour for label
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.ScalarVisibilityOff()
    
    actor = vtk.vtkActor()

    actor.SetMapper(mapper)

    # actor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
    actor.GetProperty().SetDiffuseColor(0.9000, 0.0882, 0.0784)
    # actor.GetProperty().SetColor(colors.GetColor3d('MistyRose'))
    # actor.GetProperty().SetColor(colors.GetColor3d("Banana"))
    # surfaceActor.GetProperty().SetSpecularColor(2, .1, 1)
    # surfaceActor.GetProperty().SetSpecular(.4)
    # surfaceActor.GetProperty().SetSpecularPower(50)

# %%
    renderer = vtk.vtkRenderer()
    # renderer.SetBackground(1, 1, 1)
    renderer.AddActor(actor)
    renderer.SetBackground(.3, .2, .1) 

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.AddRenderer(renderer)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Adjust the camera for front-to-back view
    camera = renderer.GetActiveCamera()
    # camera.SetPosition(0, 0, -1)
    # camera.SetFocalPoint(0, 0, 0)
    # camera.SetViewUp(0, -1, 0)
    # 
    camera.SetPosition(0, -1, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)
    renderer.ResetCamera()

    render_window.Render()
 
# %%
