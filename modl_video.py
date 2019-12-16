import matplotlib.pyplot as mpl
# mpl.use('Qt5Agg')

from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from modl.input_data.fmri.fixes import monkey_patch_nifti_image

monkey_patch_nifti_image()

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from sklearn.model_selection import train_test_split

from modl.datasets import fetch_adhd
from modl.decomposition.fmri import fMRIDictFact, rfMRIDictionaryScorer
from modl.plotting.fmri import display_maps
from modl.utils.system import get_cache_dirs

import ibc_public
import os
import glob
import nibabel as nib

n_components = 20
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

dict_init = fetch_atlas_basc_multiscale_2015()['scale444']

_package_directory = os.path.dirname(os.path.abspath(ibc_public.__file__))
mask = nib.load(os.path.join(_package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

memory = Memory(cachedir=get_cache_dirs()[0], verbose=2)

######
# This bit is for testing the script with
# one run per subject
######

# Now create a list of movie session files
movie_dir = '/home/sshankar/raiders/3mm/'
subs = sorted(os.listdir(movie_dir))

movie_arrays = []

# Create 2D masked arrays from image data and save to file for quick and easy access
for s, sub in enumerate(subs):
    if os.path.isdir(os.path.join(movie_dir, sub)):
        sub_arrays = []
        sess = sorted(os.listdir(os.path.join(movie_dir, sub)))
        ses = sess[21]
        if os.path.isdir(os.path.join(movie_dir, sub, ses)):
            movie_imgs = sorted(glob.glob(os.path.join(movie_dir, sub, ses, '*.nii.gz')))
            mi = 0
            sub_arrays.append(movie_imgs[mi])
        movie_arrays.append(sub_arrays)

import pandas as pd
mov_df = pd.DataFrame(data=movie_arrays).values

dict_fact = fMRIDictFact(smoothing_fwhm=smoothing_fwhm,
                         standardize=True,
                         high_pass=1./128,
                         t_r=2.0,
                         method=method,
                         step_size=step_size,
                         mask=mask,
                         memory=memory,
                         memory_level=2,
                         verbose=verbose,
                         n_epochs=n_epochs,
                         n_jobs=n_jobs,
                         random_state=1,
                         n_components=n_components,
                         dict_init=dict_init,
                         positive=True,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         reduction=reduction,
                         alpha=alpha,
                         )

files_ = tuple([x[0] for x in mov_df])

dict_fact.fit(files_)
