from nilearn.input_data import NiftiMasker
from fastsrm.fastsrm import FastSRM
from nilearn import image
import numpy as np
import os
from nilearn.image import new_img_like
import ibc_public
import nibabel as nib

# Specify the data and mask files
movie_img = '/volatile/sshankar/fastsrm_data/raiders/sub-01/ses-01/wrdcsub-01_ses-16_task-Raiders_acq-ap_run-04_bold.nii.gz'
_package_directory = os.path.dirname(
    os.path.abspath(ibc_public.__file__))
mask_gm = nib.load(os.path.join(
    _package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

# Specify the atlas to use
basc444 = '/volatile/sshankar/nilearn_data/basc_multiscale_2015/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale444.nii.gz'
masker = NiftiMasker(mask_img=mask_gm, dtype=np.int32).fit()
masked_atlas = masker.transform(basc444)

fast_srm = FastSRM(
    atlas=masked_atlas,
    n_components=20,
    n_jobs=1,
    n_iter=10,
    temp_dir='/tmp',
    low_ram=True, # Let's say I really have a small memory so I need low_ram mode
    aggregate="mean" # transform will return the mean of subject specific shared response
)
fast_srm.fit(movie_img)

"""
basc_im = image.load_img(basc444).get_data()

# Set up the masker object to standardize the input image and
# mask it using the group mask (mask_img)
masker = NiftiMasker(mask_img=mask_img, standardize=True, t_r=2.0).fit()
masked_img = masker.transform(movie_img)

# Save the transformed image matrix to disk
np.save('/volatile/sshankar/fastsrm_data/raiders/sub-01/ses-01/masked_img.npy', masked_img)

# Set up a masker for the atlas using the group mask so that it has the
# same number of voxels as the masked data file
atlas_masker = NiftiMasker(mask_img=mask_img).fit()

# Now, a bit of shape shifting to make the atlas compatible with
# what fastsrm.reduce_data() requires.
# 1. Add a 4th dimension to the 3D atlas. The 4th dimension will have as many
#   elements as atlas parcesl (444, in this case)
# 2. The 3D "volume" pertaining to each 4th dimension will contain 1 in the
#   "voxel" for that parcel and 0 otherwise
# 3. Apply the atlas masker set up previously to transform the new 4D atlas
#   into 2D, with n_voxel rows and n_parcel columns,
#   where n_voxel is the number of voxels in the transformed image matrix
# 4. Reduce the 2D atlas matrix to 1D by using the argmax function along the
#   column dimension. Now, the transformed atlas has n_voxel elements.
X = basc_im
n_components = len(np.unique(X)) - 1
xa, ya, za = X.shape
A = np.zeros((xa, ya, za, n_components + 1))
for c in np.unique(X)[1:].astype(int):
    X_ = np.copy(X)
    X_[X_ != c] = 0.
    X_[X_ == c] = 1.
    A[:, :, :, c] = X_
A = atlas_masker.transform(new_img_like(basc444, A))
A = np.argmax(A, axis=0)

# # Save the transformed atlas
np.save('/volatile/sshankar/fastsrm_data/raiders/atlas_masked.npy', A)

# Load the transformed files from disk
masked_img_load = np.load('/volatile/sshankar/fastsrm_data/raiders/sub-01/ses-01/run-04_masked.npy')
atlas_load = np.load('/volatile/sshankar/fastsrm_data/raiders/atlas_masked.npy')

# Try the reduce_data_single() function to reduce dimensionality
# of the image file to n_TR x n_parcel
red_dat = fastsrm.reduce_data_single(1, 1, img=masked_img_load.T, atlas=atlas_load)
np.save('/volatile/sshankar/fastsrm_data/raiders/sub-01/ses-01/reduced_img.npy', red_dat)
"""
