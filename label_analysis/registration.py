# %%
import os
import sys
import time
from functools import reduce

import networkx as nx
import ray
import SimpleITK as sitk
from dicom_utils.capestart_related import find_files_from_list
from label_analysis.geometry import LabelMapGeometry
from label_analysis.utils import is_sitk_file
from platipy.imaging import ImageVisualiser
from platipy.imaging.registration.linear import linear_registration

from label_analysis import registration_callbacks

sys.path += ["/home/ub/code"]
import itertools as il
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr
from IPython.display import clear_output
from label_analysis.helpers import *

from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from fran.utils.string import (find_file, info_from_filename, match_filenames,
                               strip_extension, strip_slicer_strings)


def plot_values(registration_method):
    global metric_values, multires_iterations
    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()


def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []


# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


# %%
class Register:
    def __init__(self, f_im, f_lm=None) -> None:
        self.lsf = sitk.LabelShapeStatisticsImageFilter()
        f_im, f_lm = self.initialize_im_lm(f_im, f_lm)
        store_attr()

    def initialize_im_lm(self, im, lm):
        im = self.load_sitk(im, dtype=sitk.sitkFloat32)
        lm = self.load_sitk(lm, dtype=sitk.sitkUInt8)
        lm, im = self.crop_center(lm, im)
        return im, lm

    def load_sitk(self, im, dtype=sitk.sitkInt32):
        if isinstance(im, Union[Path, str]):
            if dtype is not None:
                im = sitk.ReadImage(im, dtype)
            else:
                im = sitk.ReadImage(im)
        return im

    def compute_tfm(self, m_im, m_lm):
        self.set_filename(m_im)
        self.m_im, self.m_lm = self.initialize_im_lm(m_im, m_lm)
        self.m_im_t = self.register_init(self.m_im)
        self.m_im_t = self.register_full(self.m_im_t)
        self.cmp_transform = sitk.CompositeTransform(
            [self.final_transform, self.initial_transform]
        )
        self.m_lm_t = sitk.Resample(
            self.m_lm,
            self.f_im,
            self.cmp_transform,
            sitk.sitkNearestNeighbor,
            0.0,
            self.m_lm.GetPixelID(),
        )

    def crop_center(self, lm, im=None):
        orig, sz = self.get_lm_boundingbox(lm)  # self.lsf.Execute(lm)
        lm = sitk.RegionOfInterest(lm, sz, orig)
        lm.SetOrigin([0, 0, 0])
        if im:
            im = sitk.RegionOfInterest(im, sz, orig)
            im.SetOrigin([0, 0, 0])
            return lm, im
        else:
            return lm

    def set_filename(self, fn_im):
        fn_im = Path(fn_im)
        self.fname = fn_im.name

    def get_lm_boundingbox(self, lm):
        self.lsf.Execute(lm)
        lm_bb = self.lsf.GetBoundingBox(1)
        bb_orig = lm_bb[:3]
        bb_sz = lm_bb[3:]
        return bb_orig, bb_sz

    def register_init(self, m_im):
        self.initial_transform = sitk.CenteredTransformInitializer(
            self.f_im,
            m_im,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        m_im_t = sitk.Resample(
            m_im,
            self.f_im,
            self.initial_transform,
            sitk.sitkLinear,
            0.0,
            m_im.GetPixelID(),
        )
        return m_im_t

    def apply_tfm(self, im):
        im = sitk.Resample(
            im, self.f_im, self.cmp_transform, sitk.sitkLinear, 0.0, im.GetPixelID()
        )
        return im

    def register_full(self, m_im):
        self.reg_method = sitk.ImageRegistrationMethod()
        # Similarity metric settings.
        self.reg_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.reg_method.SetMetricSamplingStrategy(self.reg_method.RANDOM)
        self.reg_method.SetMetricSamplingPercentage(0.01)
        self.reg_method.SetInterpolator(sitk.sitkLinear)
        # Optimizer settings.
        self.reg_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        self.reg_method.SetOptimizerScalesFromPhysicalShift()
        # Setup for the multi-resolution framework.
        self.reg_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        self.reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        self.reg_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        self.reg_method.SetInitialTransform(self.initial_transform, inPlace=False)
        # Connect all of the observers so that we can perform plotting during registration.
        self.reg_method.AddCommand(sitk.sitkStartEvent, start_plot)
        self.reg_method.AddCommand(sitk.sitkEndEvent, end_plot)
        self.reg_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
        )
        self.reg_method.AddCommand(
            sitk.sitkIterationEvent, lambda: plot_values(self.reg_method)
        )
        self.final_transform = self.reg_method.Execute(
            sitk.Cast(self.f_im, sitk.sitkFloat32), sitk.Cast(m_im, sitk.sitkFloat32)
        )
        m_im_t = sitk.Resample(
            m_im,
            self.f_im,
            self.final_transform,
            sitk.sitkLinear,
            0.0,
            m_im.GetPixelID(),
        )
        print("Final metric value: {0}".format(self.reg_method.GetMetricValue()))
        print(
            "Optimizer's stopping condition, {0}".format(
                self.reg_method.GetOptimizerStopConditionDescription()
            )
        )
        return m_im_t

    def apply_init(self, im):
        im = sitk.Resample(
            im, self.f_im, self.initial_transform, sitk.sitkLinear, 0.0, im.GetPixelID()
        )
        return im

    def apply_full(self, im):
        im_t = sitk.Resample(
            im, self.f_im, self.final_transform, sitk.sitkLinear, 0.0, im.GetPixelID()
        )
        return im_t

    def view_fixed(self):
        view_sitk(self.f_im, self.f_lm)

    def view_moving(self):
        view_sitk(self.m_im, self.m_lm)

    def view_transformed(self):
        view_sitk(self.m_im_t, self.m_lm)


class RegisterBSpline(Register):

    def register_full(self, m_im):
        self.reg_method = sitk.ImageRegistrationMethod()
        self.reg_method.SetMetricAsCorrelation()
        self.reg_method.SetMetricSamplingStrategy(self.reg_method.RANDOM)
        self.reg_method.SetMetricSamplingPercentage(0.01)
        self.reg_method.SetInterpolator(sitk.sitkLinear)

        self.reg_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=300
        )
        self.reg_method.SetOptimizerScalesFromPhysicalShift()

        self.reg_method.SetInitialTransform(self.initial_transform, inPlace=True)

        self.reg_method.AddCommand(
            sitk.sitkStartEvent, registration_callbacks.metric_start_plot
        )
        self.reg_method.AddCommand(
            sitk.sitkEndEvent, registration_callbacks.metric_end_plot
        )
        self.reg_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            registration_callbacks.metric_update_multires_iterations,
        )
        self.reg_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: registration_callbacks.metric_plot_values(self.reg_method),
        )

        self.final_transform = self.reg_method.Execute(self.f_im, m_im)

        m_im_t = sitk.Resample(
            m_im,
            self.f_im,
            self.final_transform,
            sitk.sitkLinear,
            0.0,
            m_im.GetPixelID(),
        )
        print("Final metric value: {0}".format(self.reg_method.GetMetricValue()))
        print(
            "Optimizer's stopping condition, {0}".format(
                self.reg_method.GetOptimizerStopConditionDescription()
            )
        )
        return m_im_t


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

    R = Register(fn1_im, fn1_lm)
    R.compute_tfm(fn2_im, fn2_lm)

    R.view_fixed()
    R.view_moving()
    R.view_transformed()
# %%
    arr = sitk.GetArrayFromImage(R.m_im_t)
    m_lm = R.apply_init(R.m_lm)
    m_lm = R.apply_full(m_lm)
    view_sitk(R.m_im_t, m_lm)
# %%

    R2 = RegisterBSpline(fn1_im, fn1_lm)
    R2.f_im.GetPixelIDTypeAsString()
    R2.compute_tfm(fn3_im, fn3_lm)
# %%

    R2.view_fixed()
    R2.view_moving()
    R2.view_transformed()

# %%

# %%
    lm_t = R2.apply_init(R2.m_lm)
    lm_t = R2.apply_full(lm_t)

    lm_t = cmp(R2.m_lm)
    lm_t = sitk.Resample(
        R2.m_lm, R2.f_im, cmp, sitk.sitkLinear, 0.0, R2.m_lm.GetPixelID()
    )
    view_sitk(R2.m_im_t, lm_t)
    cmp = sitk.CompositeTransform([R2.final_transform, R2.initial_transform])
    # cmp.AddTransform(R2.final_transform)
    # cmp.AddTransform(R2.initial_transform)
# %%

    fil = sitk.LabelShapeStatisticsImageFilter()
    lm1 = sitk.ReadImage(fn1_lm, sitk.sitkUInt8)
    fil.Execute(lm1)
    lm1bb = fil.GetBoundingBox(1)
    orig = lm1bb[:3]
    sz = lm1bb[3:]
    lm1 = sitk.RegionOfInterest(lm1, sz, orig)
    lm1.SetOrigin([0, 0, 0])
# %%
    lm2 = sitk.ReadImage(fn2, sitk.sitkUInt8)
    fil.Execute(lm2)
    lm2bb = fil.GetBoundingBox(1)
# %%

    orig2 = lm2bb[:3]
    sz2 = lm2bb[3:]
    lm2 = sitk.RegionOfInterest(lm2, sz2, orig2)
    lm2.SetOrigin([0, 0, 0])
# %%
    lm2r, tfm = linear_registration(lm1, lm2)
    arr = sitk.GetArrayFromImage(lm2r)
# %%
    view_sitk(lm2, lm2, data_types=["mask", "mask"])

    view_sitk(lm2r, lm2r, data_types=["mask", "mask"])
# %%
