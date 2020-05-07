import os
import glob
import warnings

from joblib import Memory

# Import the necessary modules/functions
from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from nistats.reporting import make_glm_report

from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn import image
import nibabel as nib
import numpy as np
import pandas as pd
import ibc_public
import matplotlib.pyplot as plt

# ############################### INPUTS ######################################

TASK = 'Raiders'

SRM_PATH = os.path.join('/home/parietal/sshankar', TASK, 'fastsrm')
if not os.path.isdir(SRM_PATH):
    os.makedirs(SRM_PATH)

GROUP_LEVEL_PATH = os.path.join('/home/parietal/sshankar', TASK, 'fastsrm', 'group_level')
if not os.path.isdir(GROUP_LEVEL_PATH):
    os.makedirs(GROUP_LEVEL_PATH)

sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
sub_path = [os.path.join(SRM_PATH, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

# Specify the mask image
_package_directory = os.path.dirname(os.path.abspath(ibc_public.__file__))
mask_gm = nib.load(os.path.join(_package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

def data_parser(data_path=SRM_PATH):
    """Generate a dataframe that contains all the data corresponding
    to the acquisitions of the raiders task."""
    paths = []
    subjects = []
    task = TASK

    for sbj in SUBJECTS:
        # Basis lists as spatial maps
        bl_name = 'basis_list*.nii.gz'
        bl_path = os.path.join(SRM_PATH, sbj, bl_name)
        bl = glob.glob(os.path.join(SRM_PATH, sbj, bl_name))

        if not bl:
            msg = 'basis_list*.nii.gz file for task ' + \
                  '%s in %s not found!' % (task, sbj)
            warnings.warn(msg)

        path_parts = os.path.split(sbj)
        for part in path_parts:
            if part[0:3] == 'sub':
                subject = part

        for img in bl:
            paths.append(img)
            subjects.append(subject)

    # create a dictionary with all the information
    db_dict = dict(
        path=paths,
        subject=subjects,
    )
    # create a DataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db

def do_second_level(second_level_input, smoothing_fwhm=8, n_comp=20):
    # Construct a design matrix. We are including all subjects and
    # essentially finding the "main effects" of the contrasts performed
    # in the first level analysis
    design_matrix = pd.DataFrame([1] * len(second_level_input), columns=['intercept'])
    # Set up the second level analysis
    second_level_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm)

    # Compute the contrast/main effect
    for i in range(n_comp):
        z_map = second_level_model.fit(list(second_level_input[:,i]),
                                        design_matrix=design_matrix).compute_contrast(output_type='z_score')
        nib.save(new_img_like(mask_gm, z_map.get_fdata()),
                 os.path.join(GROUP_LEVEL_PATH, 'component-%02d.nii.gz' %i))
        report = make_glm_report(second_level_model, 'intercept')
        report.save_as_html(os.path.join(GROUP_LEVEL_PATH, 'component-%02d.html' % i))

if __name__ == '__main__':
    db = data_parser(SRM_PATH)
    n_comp = len(db[db.subject == 'sub-01'].path)
    smoothing_fwhm = 8

    second_level_input = np.empty((len(sub_no),n_comp), dtype='object')

    for s, subject in enumerate(SUBJECTS):
        # Let's gather the files that will form the input to the second level analysis
        second_level_input[s] = db[db.subject == subject].path

    do_second_level(second_level_input, smoothing_fwhm, n_comp)
