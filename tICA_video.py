from sklearn.decomposition import FastICA
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import ibc_public
import os

# Load input data
task = 'raiders'
datapath = '../videos_analysis'
sub_movie = np.load(os.path.join(datapath, task + '_concat_data.npy'), allow_pickle=True)

# Set up some parameters for the ICA
n_components = 20
random_state = 0

# Initialize the ICA model
fast_ica = FastICA(n_components=n_components,
                  random_state=random_state)

# Transform input data using the ICA model
data_transform = fast_ica.fit_transform(sub_movie.T).T
np.save(os.path.join(datadir, 'fastica_components.npy'), data_transform)
