# encoding: utf-8

"""
Here  we show all the helper functions..

"""

import SimpleITK as sitk
import numpy as np
import pandas as pd


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def prep_splitcol(pd_x, new_col, orig_col, splt_key=r'\s'):
    # Simple prep for a specific column..

    pd_x[new_col] = pd_x[orig_col].str.strip().str.split(splt_key, expand=True)
    pd_x[new_col] = pd_x[new_col].apply(pd.to_numeric, errors='coerce', axis=1)
    return pd_x
