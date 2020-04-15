from nilearn import image
import numpy as np
import glob
import os

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

data_dir = '/storage/store/data/ibc/derivatives/'

# Which data file to use to compute confounds
filepattern = 'dc*bold.nii.gz'

for s, sub in enumerate(subs_hi):
    sess = sorted(glob.glob(sub + '/ses*'))
    for si, ses in enumerate(sess):

subs = sorted(glob.glob(data_dir + 'sub*'))
len_hi = np.zeros((len(subs), 2, 6))
for s, sub in enumerate(subs):
    if len(final_sess[s]) > 0:
        # Get data from the sessions in final_sess
        for si, ses in enumerate(final_sess[s]):
            nii_files = sorted(glob.glob(os.path.join(sub, ses, 'func/') + filepattern))
            for mi, movie_img in enumerate(nii_files):
                dat = image.load_img(movie_img)
                len_hi[s,si,mi] = dat.shape[-1]

print(len_hi)
