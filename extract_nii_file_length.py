from nilearn import image
import numpy as np
import glob
import os

home = '/home/parietal/sshankar'
ibc_folder = 'ibc'

data_dir = '/storage/store/data/ibc/derivatives/'

# Which data file to use to compute confounds
filepattern = 'dc*bold.nii.gz'

len_hi = np.zeros((len(subs_hi), 2, 6))

for s, sub in enumerate(subs_hi):
    sess = sorted(glob.glob(sub + '/ses*'))
    for si, ses in enumerate(sess):
        nii_files = sorted(glob.glob(ses + '/' + '*nii.gz'))
        for mi, movie_img in enumerate(nii_files):
            dat = image.load_img(movie_img)
            len_hi[s,si,mi] = dat.shape[-1]

print(len_hi)
