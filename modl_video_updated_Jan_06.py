# This script uses the MODL model to compute brain maps that represent
# common processing during viewing of video clips.

# Created: January 2020
# Author: Swetha Shankar

import matplotlib.pyplot as mpl
from nilearn.datasets import fetch_atlas_smith_2009
from modl.input_data.fmri.fixes import monkey_patch_nifti_image
from sklearn.externals.joblib import Memory
from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs
import ibc_public
import os
import glob
import nibabel as nib
import pandas as pd

# Change some default fmri file-related functions to custom ones
monkey_patch_nifti_image()

# Sepcify some input paths/folders
home = '/home/parietal/sshankar'
movie_dir = '/storage/store/data/ibc/3mm/'

# Task of interest
task = 'clips'

# List the specific subjects/sessions/files that should be used
if task == 'clips':
    filepattern = '*Trn*.nii.gz'
    subs = ['sub-01', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08',
            'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']
    sessions = [['ses-06', 'ses-08', 'ses-09', 'ses-11'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-09'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-09'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-06', 'ses-07', 'ses-08', 'ses-10'],
                ['ses-04', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-05', 'ses-06', 'ses-07', 'ses-08'],
                ['ses-07', 'ses-08', 'ses-09', 'ses-10'],
                ['ses-07', 'ses-08', 'ses-09', 'ses-11']]
else:
    filepattern = '*.nii.gz'
    subs = ['sub-01', 'sub-04', 'sub-05', 'sub-06', 'sub-07',
            'sub-08', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
    sessions = [['ses-16', 'ses-17'],
                ['ses-13', 'ses-14'],
                ['ses-13', 'ses-14'],
                ['ses-13', 'ses-14'],
                ['ses-14', 'ses-15'],
                ['ses-14', 'ses-15'],
                ['ses-13', 'ses-15'],
                ['ses-13', 'ses-15'],
                ['ses-13', 'ses-14'],
                ['ses-14', 'ses-15']]

# Now create a list of movie session files
movie_arrays = []

# Create 2D masked arrays from image data and save to file for quick and easy access
for s, sub in enumerate(subs):
    if os.path.isdir(os.path.join(movie_dir, sub)):
        # sess = sorted(glob.glob(os.path.join(movie_dir, sub) + '/ses*'))
        for i, ses in enumerate(sessions):
            if os.path.isdir(os.path.join(movie_dir, sub, ses)):
                movie_imgs = sorted(glob.glob(os.path.join(movie_dir, sub, ses) + '/func/' + filepattern))
                for mi, mimg in enumerate(movie_imgs):
                    movie_arrays.append(mimg)

# Some manipulations to get the input in an acceptable format for MODL
mov_df = pd.DataFrame(data=movie_arrays).values
print(mov_df)
files_ = tuple([x[0] for x in mov_df])

# Set up some parameters for MODL
n_components = [20, 50, 100, 200]
batch_size = 50
learning_rate = .92
method = 'masked'
step_size = 0.01
reduction = 12
alpha = 1e-3
n_epochs = 2
verbose = 15
n_jobs = 2
smoothing_fwhm = 6
memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

dict_init = fetch_atlas_smith_2009().rsn20

# Use the IBC group mask for masking data
_package_directory = os.path.dirname(os.path.abspath(ibc_public.__file__))
mask = nib.load(os.path.join(_package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

# dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
#                          standardize=True,
#                          high_pass=1./128,
#                          t_r=2.0,
#                          method=method,
#                          step_size=step_size,
#                          mask=mask,
#                          memory_level=2,
#                          verbose=verbose,
#                          n_epochs=n_epochs,
#                          n_jobs=n_jobs,
#                          random_state=1,
#                          n_components=n_components,
#                          dict_init=dict_init,
#                          positive=True,
#                          learning_rate=learning_rate,
#                          batch_size=batch_size,
#                          reduction=reduction,
#                          alpha=alpha,
#                          )
#                          # memory=memory,
#
# dict_fact.fit(files_)
