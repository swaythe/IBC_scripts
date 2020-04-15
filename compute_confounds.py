from nilearn.image import high_variance_confounds
from nilearn import image, plotting
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

home = '/home/parietal/sshankar'
ibc_folder = 'ibc'

# The sessions.csv file contains information on which task
# was acquired in which session
# Use it to also get the list of subjects and sessions
sessfile = os.path.join(home, ibc_folder, 'public_analysis_code/ibc_data/sessions.csv')
sess_df = pd.read_csv(sessfile)
subjects = sess_df.subject
sessions = sess_df.columns[1:]

# What is the task of interest?
task = 'raiders'

# For each subject, find which sessions contain the task of interest
task_sess = []

for i in range(len(subjects)):
#     if subjects[i] in subs:
    ser = sess_df.iloc[i,:]
    ids = ser.str.contains(task)==True
    sess = ser.loc[ids].keys().tolist()
    task_sess.append(sess)

# Some subjects have an extra session. If so, ignore the first session.
# Number of sessions per task
if task=='raiders':
    nsess = 2
elif task=='clips':
    nsess = 3

final_sess = []
for s in range(len(task_sess)):
    if len(task_sess[s]) > nsess:
        l = len(task_sess[s])
        final_sess.append(task_sess[s][l-nsess:])
    else:
        final_sess.append(task_sess[s])

# Some useful directories
dir_local = os.path.join(home, task, '3mm')
dir_drago = '/storage/store/data/ibc/derivatives/'

# Which data file to use to compute confounds
filepattern = 'wrdc*.nii.gz'

# Now create a list of movie session files
movie_dir = os.path.join(home, task, 'derivatives/')
confound_file = 'confounds_run' +  str(run).zfill(2) + '.npy'

# Calculate high variance confounds for the data files
subs = sorted(glob.glob(movie_dir + 'sub*'))
for s, sub in enumerate(subs):
    # Get data from the sessions in final_sess
    for si, ses in enumerate(final_sess):
        movie_imgs = sorted(glob.glob(os.path.join(sub, ses) + '/' + filepattern))
        print(movie_imgs)
        # for mi, movie_img in enumerate(movie_imgs):
        #     if os.path.isfile(movie_img) and not os.path.isfile(os.path.join(ses, confound_file)):
        #         movie_imgs_confounds = high_variance_confounds(movie_img)
        #         np.save(os.path.join(ses, confound_file), movie_imgs_confounds)
        #     else:
        #         print('Confounds file already exists')
