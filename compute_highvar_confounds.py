import os
import glob
import warnings

from joblib import Memory

from nilearn.image import high_variance_confounds
from nilearn import image, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ############################### INPUTS ######################################

DERIVATIVES = '/storage/store/data/ibc/3mm'

sub_no = [1, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]
task = 'Raiders'

sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

mem = Memory(cachedir=cache, verbose=0)

def data_parser(derivatives=DERIVATIVES):
    """Generate a dataframe that contains all the data corresponding
    to the acquisitions of the raiders task."""
    paths = []
    subjects = []
    sessions = []
    acquisitions = []

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
                        task = part[5:]
                    elif part[:4] == 'dir-':
                        acquisition = part[4:]
                if task not in tasks:
                    continue
                paths.append(img)
                sessions.append(session)
                subjects.append(subject)
                acquisitions.append(acquisition)
                print('task: %s, sub: %s, ses: %s, acq: %s', task, subject, session, acquisition)

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

if __name__ == '__main__':
    db = data_parser(derivatives=DERIVATIVES)

    # per-subject high-variance confounds
    for subject in SUBJECTS:
        # Calculate high variance confounds for the data files
        print(subject)
        # for si, ses in enumerate(sess):
        #     movie_imgs = sorted(glob.glob(ses + '/' + filepattern))
        #     for mi, movie_img in enumerate(movie_imgs):
        #         if os.path.isfile(movie_img) and not os.path.isfile(os.path.join(ses, confound_file)):
        #             movie_imgs_confounds = high_variance_confounds(movie_img)
        #             np.save(os.path.join(ses, confound_file), movie_imgs_confounds)
