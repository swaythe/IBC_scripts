import os
import glob
import warnings

from joblib import Memory

from fastsrm.identifiable_srm import IdentifiableFastSRM
from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
import nibabel as nib
import numpy as np
import pandas as pd
import ibc_public
import matplotlib.pyplot as plt

# ############################### INPUTS ######################################

TASK = 'Raiders'

PREPROC_PATH = os.path.join('/home/parietal/sshankar', TASK, 'preproc')
ATLAS_PATH = '/home/parietal/sshankar/basc'

SRM_PATH = os.path.join('/home/parietal/sshankar', TASK, 'fastsrm')
if not os.path.isdir(SRM_PATH):
    os.makedirs(SRM_PATH)

sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
sub_path = [os.path.join(PREPROC_PATH, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

SESSIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Specify the mask image
_package_directory = os.path.dirname(os.path.abspath(ibc_public.__file__))
mask_gm = nib.load(os.path.join(_package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

def data_parser(data_path=PREPROC_PATH):
    """Generate a dataframe that contains all the data corresponding
    to the acquisitions of the raiders task."""
    paths = []
    subjects = []
    task = TASK

    for sbj in SUBJECTS:
        # Preprocessed files converted to 2D (nVoxel X nTR) arrays
        for ses in SESSIONS:
            npy_name = 'preproc%s_ses-*_task-%s*_run-%02d_bold.npy' \
                        % (sbj, task, ses)
            npy_path = os.path.join(data_path, 'sub-*',
                                     npy_name)
            npy = glob.glob(npy_path)
            if not npy:
                msg = 'preproc*.npy file for task ' + \
                      '%s %s in %s not found!' % (task, acq, sbj)
                warnings.warn(msg)

            # If multiple recordings were made of the same session, use the last one
            if len(npy) > 1:
                npy = [npy[-1]]

            basename = os.path.basename(npy[0])
            parts = basename.split('_')
            print(parts)
            task_ = None
            for part in parts:
                if part[7:10] == 'sub':
                    subject = part[7:13]
                    print(subject)
                elif part[:5] == 'task-':
                    task_ = part[5:]
            if task_ not in TASK:
                continue
            paths.append(npy)
            subjects.append(subject)

    # create a dictionary with all the information
    db_dict = dict(
        path=paths,
        subject=subjects,
    )
    # create a DataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db

def get_transformed_atlas():
    """Get a transformed version of the atlas being used because the
    FastSRM algorithm doesn't work with a file name right now."""
    # Do this for a previously unused atlas.
    # Else, you should have a .npy file saved from before, and you can just load it.
    # The transform() funtion takes a few minutes to run so don't run it
    # unless you absolutely need to.

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

    if os.path.exists(os.path.join(ATLAS_PATH, 'atlas_masked.npy')):
        atlas = np.load(os.path.join(ATLAS_PATH, 'atlas_masked.npy'), allow_pickle=True)
    else:
        # Specify the atlas
        basc444 = fetch_atlas_basc_multiscale_2015()['scale444']
        basc_im = image.load_img(basc444).get_data()

        atlas_masker = NiftiMasker(mask_img=mask_gm).fit()

        if len(basc_im.shape) == 3:
            n_components = len(np.unique(basc_im)) - 1
            xa, ya, za = basc_im.shape
            A = np.zeros((xa, ya, za, n_components + 1))
            atlas = np.zeros((xa, ya, za, n_components + 1))
            for c in np.unique(basc_im)[1:].astype(int):
                X_ = np.copy(basc_im)
                X_[X_ != c] = 0.
                X_[X_ == c] = 1.
                A[:, :, :, c] = X_
            atlas = atlas_masker.transform(new_img_like(basc444, A))
            atlas = np.argmax(atlas, axis=0)

        # Save the transformed atlas
        if not os.path.exists(ATLAS_PATH):
            os.makedirs(ATLAS_PATH)
        np.save(os.path.join(ATLAS_PATH, 'atlas_masked.npy'), atlas)

        return atlas

def apply_fastsrm(db, atlas, n_comp=20, n_jobs=1, n_iter=10, tmp='/home/sshankar/parietal/tmp'):
    # Fit the FastSRM model with the data
    fast_srm = FastSRM(
        atlas=atlas,
        n_components=n_comp,
        n_jobs=n_jobs,
        n_iter=n_iter,
        temp_dir=tmp,
        low_ram=True,
        aggregate="mean",
    )
    fast_srm.fit(srm_data)
    shared_resp = fast_srm.transform(srm_data)

    # Plot the shared responses
    fig, axs = plt.subplots(5, sharex=True, sharey=True, figsize=(10,50))
    for i in range(len(shared_resp)):
        axs[i].plot(shared_resp[i,:])
        axs[i].set_title('Shared response #' + str(i+1))

    # Save the shared response matrix and figure
    np.save(os.path.join(SRM_PATH, '%s_shared-responses.npy' %task), shared_resp)
    fig.savefig(os.path.join(SRM_PATH, '%s_shared-responses.pdf' %task), format='pdf', transparent=False)
    save_basis_functions(fast_srm.basis_list, n_comp)

def save_basis_functions(basis_list, n_comp):
    # Save the basis lists to subject folders for posterity
    for s, subject in enumerate(SUBJECTS):
        bl_ = basis_list[s]
        copyfile(bl_, os.path.join(SRM_PATH, subject, 'basis_list.npy'))
        for i in range(n_comp):
            nib.save(img_masker.inverse_transform(bls[i]),
                os.path.join(SRM_PATH, subject, 'basis_list-' + str(i).zfill(2) + '.nii.gz'))

if __name__ == '__main__':
    db = data_parser(PREPROC_PATH)

    # Specify FastSRM parameters
    atlas = get_transformed_atlas()
    n_comp = 20
    n_jobs = 1
    n_iter = 10
    tmp = '/home/sshankar/parietal/tmp'

    data = []

    for subject in SUBJECTS:
        print(subject)
        data_ = []
        data_files = db[db.subject == subject].path
        for dfi, df in enumerate(data_files):
            print(df)
            data_.append(np.load(df, allow_pickle=True))
        data.append(np.concatenate(data_))
        print(len(data), data[0].shape)
    # apply_fastsrm(data, atlas, n_comp, n_jobs, n_iter, tmp)
