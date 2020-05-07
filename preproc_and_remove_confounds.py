import os
import glob
import warnings

from joblib import Memory

from nilearn.input_data import NiftiMasker
import nibabel as nib
import numpy as np
import pandas as pd
import ibc_public

# ############################### INPUTS ######################################

DERIVATIVES = '/storage/store/data/ibc/3mm'

sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
TASK = 'Raiders'

CONFOUND_PATH = os.path.join('/home/parietal/sshankar', TASK, 'confounds')
if not os.path.isdir(CONFOUND_PATH):
    os.makedirs(CONFOUND_PATH)

PREPROC_PATH = os.path.join('/home/parietal/sshankar', TASK, 'preproc')
if not os.path.isdir(PREPROC_PATH):
    os.makedirs(PREPROC_PATH)

sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

# Specify the mask image
_package_directory = os.path.dirname(os.path.abspath(ibc_public.__file__))
mask_gm = nib.load(os.path.join(_package_directory, '../ibc_data', 'gm_mask_3mm.nii.gz'))

def data_parser(derivatives=DERIVATIVES):
    """Generate a dataframe that contains all the data corresponding
    to the acquisitions of the raiders task."""
    paths = []
    subjects = []
    sessions = []
    acquisitions = []
    task = TASK

    for sbj in SUBJECTS:
        # fixed-effects activation images
        for acq in ['pa', 'ap']:
            bold_name = 'wrdc%s_ses*_task-%s_dir-%s*_bold.nii.gz' \
                        % (sbj, task, acq)
            bold_path = os.path.join(derivatives, 'sub-*/ses-*/func',
                                     bold_name)
            bold = glob.glob(bold_path)
            if not bold:
                msg = 'wrdc*.nii.gz file for task ' + \
                      '%s %s in %s not found!' % (task, acq, sbj)
                warnings.warn(msg)

            for img in bold:
                basename = os.path.basename(img)
                parts = basename.split('_')
                task_ = None
                for part in parts:
                    if part[4:7] == 'sub':
                        subject = part[4:10]
                    elif part[:3] == 'ses':
                        session = part
                    elif part[:5] == 'task-':
                        task_ = part[5:]
                    elif part[:4] == 'dir-':
                        acquisition = part[4:]
                if task_ not in TASK:
                    continue
                paths.append(img)
                sessions.append(session)
                subjects.append(subject)
                acquisitions.append(acquisition)

    # create a dictionary with all the information
    db_dict = dict(
        path=paths,
        subject=subjects,
        session=sessions,
        acquisition=acquisitions,
    )
    # create a DataFrame out of the dictionary and write it to disk
    db = pd.DataFrame().from_dict(db_dict)
    return db

def preprocess(df, confound_file, preproc_file):
    # Get the part of the file name that has task, sub, ses and acq info
    df_name = os.path.split(df)[1]
    temp_name = (df_name.split('.'))[0][4:]

    # Set up masketer
    masker = NiftiMasker(mask_img=mask_gm,
                                standardize=True,
                                smoothing_fwhm=5,
                                detrend=True,
                                high_pass=1./128,
                                t_r=2.0).fit()

    # Apply masker and remove high variance confounds
    preproc_array_ = masker.transform(df, confounds=confound_file).T
    np.save(os.path.join(PREPROC_PATH, subject, preproc_file), preproc_array_)

if __name__ == '__main__':
    db = data_parser(derivatives=DERIVATIVES)
    nconf = 5

    # per-subject preprocessing and high-variance confounds removal
    for subject in SUBJECTS:
        print(subject)
        # Calculate high variance confounds for the data files
        # conf_files = []
        data_files = db[db.subject == subject].path
        for dfi, df in enumerate(data_files):
            # Get the part of the file name that has task, sub, ses and acq info
            df_name = os.path.split(df)[1]
            temp_name = (df_name.split('.'))[0][4:]

            if not os.path.isdir(os.path.join(PREPROC_PATH, subject)):
                os.makedirs(os.path.join(PREPROC_PATH, subject))
            preproc_file = os.path.join(PREPROC_PATH, subject, 'preproc%s.npy' %temp_name)

            confound_file = os.path.join(CONFOUND_PATH, 'conf%s.csv' %temp_name)

            preprocess(df, confound_file, preproc_file)
