# Assembled from Jupyter notebook: November 08, 2019
# Author: SS

import os
from numpy import zeros
from pathlib import Path

# How many sessions was this data collected in?
n_ses = 2

# Define subjects
# subs = ['sub-01', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
subs = ['sub-01', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-09', 'sub-13', 'sub-14']

# Create as many session arrays as required
ses = zeros((n_ses, len(subs)), dtype=int)
# ses[0] = [16, 13, 13, 13, 14, 14, 14, 14, 13, 14]
# ses[1] = [17, 14, 14, 14, 15, 15, 15, 15, 14, 15]
ses[0] = [16, 13, 13, 13, 14, 14, 13, 14]
ses[1] = [17, 14, 14, 14, 15, 15, 14, 15]

# Specify the local and remote directories
local_dir = Path('/volatile/sshankar/fastsrm_data/raiders/3mm/')
remote_dir = Path('sshankar@drago:/storage/store/data/ibc/3mm/')

# Create directories for subjects and sessions and copy files
for s, sub in enumerate(subs):
    for i in range(n_ses):
        ses_path = Path.joinpath(local_dir, sub, 'ses-' + str(i+1).zfill(2))

        # Create session directory if it doesn't exist
        if not ses_path.exists():
            os.makedirs(ses_path)

        # Copy files
        os.chdir(ses_path)
        remotefile = Path.joinpath(remote_dir, sub, 'ses-' + str(ses[i,s]), 'func/wrd*')
        os.system('scp %s .' % (remotefile))
