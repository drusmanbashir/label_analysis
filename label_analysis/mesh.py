# %%
import itk
import vtk
if __name__ == "__main__":

# %%
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

    ImageType = itk.Image[itk.UC,3]
    MeshType = itk.Mesh[itk.SS,3]
    reader = itk.ImageFileReader[ImageType].New(FileName = fn1_lm)

# %%

    # threshold = vtk.vtkImageThreshold ()
    threshold = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    threshold.SetInput(reader.GetOutput())
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.SetInput(reader.GetOutput());
    threshold.SetLowerThreshold(1);
    threshold.SetUpperThreshold(1);
    threshold.SetOutsideValue(0);


    fil = itk.BinaryMask3DMeshSource(itk.Image[itk.UC,3], itk.Image[itk.SS,3])
# %%

    img = reader.GetOutput()
    img.GetImageDimension()

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutput())
    dmc.GenerateValues(1, 1, 1)
    # dmc.SetInputData??
    dmc.Update()




# %%
