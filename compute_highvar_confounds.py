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
TASK = 'Raiders'

CONFOUND_PATH = os.path.join('/home/parietal/sshankar', TASK, 'confounds')
if not os.path.isdir(CONFOUND_PATH):
    os.makedirs(CONFOUND_PATH)

FIG_PATH = os.path.join('/home/parietal/sshankar', TASK, 'confounds', 'figures')
if not os.path.isdir(FIG_PATH):
    os.makedirs(FIG_PATH)

sub_path = [os.path.join(DERIVATIVES, 'sub-%02d' % s) for s in sub_no]
SUBJECTS = [os.path.basename(full_path) for full_path in sub_path]

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

def compute_confound(df, nconf, confound_file):
    # Compute high variance confounds and save file
    movie_imgs_confounds = high_variance_confounds(df, n_confounds=nconf)
    np.savetxt(os.path.join(CONFOUND_PATH, confound_file), movie_imgs_confounds)

def make_confound_fig(conf_files, nconf, fig_file):
    nses = len(conf_files)

    # Plot confounds for all sessions of the subject
    fig, axs = plt.subplots(nses, nconf, figsize=(nconf*3,10))
    plt.subplots_adjust(hspace=1)

    # Iterate through all sessions
    for cfi,cf in enumerate(conf_files):
        # Load the confound files
        ses = cf.split('_')
        confounds = np.loadtxt(os.path.join(CONFOUND_PATH, cf))
        axs[0,round(nconf/2)].set_title(ses[0][4:])
        for c in range(nconf):
            axs[cfi,c].plot(confounds[:,c], 'b-')
            if c==0:
                axs[cfi,c].set_ylabel(ses[-2])
            axs[cfi,c].set_xticklabels(labels=[])
            axs[cfi,c].set_yticklabels(labels=[])

    fig.savefig(os.path.join(FIG_PATH, fig_file), format='pdf', transparent=False)

if __name__ == '__main__':
    db = data_parser(derivatives=DERIVATIVES)
    nconf = 5

    # per-subject high-variance confounds
    for subject in SUBJECTS:
        print(subject)
        # Calculate high variance confounds for the data files
        conf_files = []
        data_files = db[db.subject == subject].path
        for dfi, df in enumerate(data_files):
            # Get the part of the file name that has task, sub, ses and acq info
            df_name = os.path.split(df)[1]
            temp_name = (df_name.split('.'))[0][4:]
            confound_file = 'conf%s.csv' %temp_name
            conf_files.append(confound_file)
            compute_confound(df, nconf, confound_file)

        fig_file = '%s.pdf' %subject
        make_confound_fig(conf_files, nconf, fig_file)

