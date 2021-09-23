"""
Author: Bertrand Thirion, 2015
Modified by: Swetha Shankar, 2021
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ibc_public.utils_data import get_subject_session
from joblib import Memory, Parallel, delayed

source_dir = '/home/sshankar/diffusion/sourcedata'
derivatives_dir = '/home/sshankar/diffusion/derivatives'
subjects_sessions = [('sub-04', 'ses-08')]

def concat_images(in_imgs, out_img):
    nib.nifti1.save(nib.funcs.concat_images(in_imgs, axis=3), out_img)

def concat_bvals(in_bvals, out_bvals):
    bvals_ = np.loadtxt(in_bvals[0], dtype=int)
    for i in range(len(in_bvals)-1):
        bv = np.loadtxt(in_bvals[i+1], dtype=int)
        bvals_ = np.concatenate((bvals_,bv))
    np.savetxt(out_bvals, bvals_, fmt='%d', newline=' ')

def concat_bvecs(in_bvecs, out_bvecs):
    bvecs_ = np.loadtxt(in_bvecs[0])
    for i in range(len(in_bvecs)-1):
        bv = np.loadtxt(in_bvecs[i+1])
        bvecs_ = np.concatenate((bvecs_,bv), axis=1)
    np.savetxt(out_bvecs, bvecs_)

def denoise_dwi(in_dn, out_dn):
    cmd = 'dwidenoise %s %s' % (in_dn, out_dn)
    print(cmd)
    os.system(cmd)

def degibbs_dwi(in_dg, out_dg):
    cmd = 'mrdegibbs %s %s' % (in_dg, out_dg)
    print(cmd)
    os.system(cmd)

def collate_b0s(b0_imgs, vols, merged_b0_img):
    cmd = "fslroi %s %s %d 1" % (b0_imgs, merged_b0_img, vols[0])
    print(cmd)
    os.system(cmd)
    cmd = "fslroi %s temp_vol %d 1" % (b0_imgs, vols[1])
    print(cmd)
    os.system(cmd)
    cmd = "fslmerge -t %s %s temp_vol" % (merged_b0_img, merged_b0_img)
    print(cmd)
    os.system(cmd)
    cmd = "fslroi %s temp_vol %d 1" % (b0_imgs, vols[2])
    print(cmd)
    os.system(cmd)
    cmd = "fslmerge -t %s %s temp_vol" % (merged_b0_img, merged_b0_img)
    print(cmd)
    os.system(cmd)
    cmd = "fslroi %s temp_vol %d 1" % (b0_imgs, vols[0])
    print(cmd)
    os.system(cmd)
    cmd = "fslmerge -t %s %s temp_vol" % (merged_b0_img, merged_b0_img)
    print(cmd)
    os.system(cmd)
    cmd = "rm temp_vol*"
    print(cmd)
    os.system(cmd)

def calc_topup(merged_b0_img, acq_params_file, hifi_file, topup_results_basename):
    cmd = "topup --imain=%s --datain=%s --config=b02b0.cnf --out=%s --iout=%s" % (
        merged_b0_img, acq_params_file, topup_results_basename, hifi_file)
    print(cmd)
    os.system(cmd)

def make_hifi_mask(hifi_file, threshold, hifi_brain):
    cmd = "fslmaths %s -Tmean temp" % (hifi_file)
    print(cmd)
    os.system(cmd)

    cmd = "bet temp %s -f %f -R -m" % (hifi_brain, threshold)
    print(cmd)
    os.system(cmd)

def make_acqdir_file(index_file, nvols)
    inds = np.concatenate((np.ones(nvols*2, dtype=int), np.ones(nvols*2, dtype=int)*2))
    np.savetxt(index_file, inds, fmt='%d')

def run_eddy(eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out):
    cmd = "eddy --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --repol --out=%s" % (
        eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out)

    print(cmd)
    os.system(cmd)

def extract_and_mask_eddy_b0(eddy_out, b0_vol, b0_mask):
    cmd = "fslroi %s %s 0 1" % (eddy_out, b0_vol)
    print(cmd)
    os.system(cmd)
    # Masking TBD

def bias_correct():
    # Correct for negative values and values close to 0 prior to bias correction
    # For more details see ANTs documentation for N4BiasFieldCorrection
    # eddy_out = glob.glob('%s/eddy_denoise.nii.gz' % dwi_dir)[0]
    # img_ = nib.load(eddy_out)
    # data_ = img_.get_fdata()
    # min_val = np.min(np.min(np.min(data_)))
    # data_ = data_ + np.abs(min_val) + (0.1 * np.abs(min_val))
    #
    # nonneg_file = os.path.join(dwi_dir, 'nn_eddy_denoise.nii.gz')
    # nib.save(nib.Nifti1Image(data_, img_.affine), nonneg_file)
    eddy_out = glob.glob('%s/nn_eddy_denoise.nii.gz' % dwi_dir)[0]
    bf_out = os.path.join(dwi_dir, 'ants_bf_nn_eddy_denoise.nii.gz')
    # cmd = "dwibiascorrect ants %s %s -mask %s -fslgrad %s %s" % (
    #     eddy_out, bf_out, mask_img, out_bvecs, out_bvals)
    # cmd = "N4BiasFieldCorrection -d 4 -i %s -w %s -o %s -s 2 -b [150] -c [200x200,0.0]" % (
    #     eddy_out, mask_img, bf_out)
    # print(cmd)
    # os.system(cmd)

def run_dmri_preproc(subject_session):
    subject, session = subject_session

    src_dir = os.path.join(source_dir, subject, session)
    dest_dir = os.path.join(derivatives_dir, subject, session)

    src_anat_dir = os.path.join(src_dir, 'anat')
    src_dwi_dir = os.path.join(src_dir, 'dwi')
    src_fmap_dir = os.path.join(src_dir, 'fmap')

    dest_dwi_dir = os.path.join(dest_dir, 'dwi')
    dest_fmap_dir = os.path.join(dest_dir, 'fmap')
    dest_anat_dir = os.path.join(dest_dir, 'anat')

    # Concatenate all DWI images (4 runs in our case)
    dwi_imgs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.nii.gz')))
    out_concat = os.path.join(dest_dwi_dir, '%s_%s_dwi.nii.gz' % (subject, session))
    concat_images(dwi_imgs, out_concat)

    # Concatenate the bval and bvec files as well
    in_bvals = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bval')))
    out_bvals = os.path.join(dest_dwi_dir, "bvals")
    concat_bvals(in_bvals, out_bvals)

    in_bvecs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bvec')))
    out_bvecs = os.path.join(dest_dwi_dir, "bvecs")
    concat_bvecs(in_bvecs, out_bvecs)

    # Denoise images using MP-PCA
    out_dn = os.path.join(dest_dwi_dir, 'dn_%s_%s_dwi.nii.gz' % (subject, session))
    denoise_dwi(out_concat, out_dn)

    # Remove Gibbs ringing artifacts
    # The recommendation is to do this if not using partial Fourier acquisition, but we do use it.
    # The images look a little blurred and of lower intensity than the denoised images
    # out_dg = os.path.join(dest_dwi_dir, 'dg_%s_%s_dwi.nii.gz' % (subject, session))
    # degibbs_dwi(out_dn, out_dg)

    # Run FSL topup - it's a 2-step process
    # 1. Collect all the b=0 volumes in one file and use that as input to topup
    b0_imgs = sorted(glob.glob('%s/dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session)))[0]
    merged_b0_img = os.path.join(dest_dwi_dir, 'b0s_%s_%s_dwi.nii.gz' % (subject, session))
    vols = [0, 61, 122, 183]
    collate_b0s(b0_imgs, vols, merged_b0_img)

    # 2. Calculate distortion from the collated b0 images
    acq_params_file = os.path.join(src_dwi_dir, 'b0_acquisition_params.txt')
    topup_results_basename = os.path.join(dest_dwi_dir, '%s_%s_topup-results' % (subject, session))
    hifi_file = os.path.join(dest_dwi_dir, '%s_%s_hifi-b0' % (subject, session))
    calc_topup(merged_b0_img, acq_params_file, hifi_file, topup_results_basename)

    # Create a text file that contains, for each volume in the concatenated dwi images file,
    # the corresponding line of the acquisition parameters file.
    # The way the data has been concatenated, we have 2 AP runs followed by 2 PA runs,
    # each with 61 volumes. Thus, the text file will have 244 lines, the first 122 will
    # say "1" and the last 122 will say "2"
    index_file = os.path.join(dest_dwi_dir, 'dwi_acqdir_index.txt')
    nvols = 61
    make_acqdir_file(index_file, nvols)

    # Now run eddy to correct eddy current distortions
    mask_img = glob.glob('%s/*mask.nii.gz' % dest_dwi_dir)[0]
    eddy_in = out_dn
    eddy_out = os.path.join(dest_dwi_dir, 'eddy_dn_%s_%s_dwi.nii.gz' % (subject, session))
    run_eddy(eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out)

    # Once again extract the b0 volumes, this time from the eddy corrected images,
    # create a mean volume, and a mask of the mean volume
    b0_imgs = glob.glob('%s/eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    merged_b0_img = os.path.join(dest_dwi_dir, 'b0s_eddy_dn_%s_%s_dwi.nii.gz' % (subject, session))
    vols = [0, 61, 122, 183]
    collate_b0s(b0_imgs, vols, merged_b0_img)
    b0_brain = os.path.join(dest_dwi_dir, 'b0_brain_eddy_dn_%s_%s_dwi' % (subject, session))
    make_hifi_mask(merged_b0_img, threshold, b0_brain)

    # Bias field correction
    # Bias field correction doesn't work very well via dwibiascorrect, and
    # fails when I try running N4BiasFieldCorrection. Skipping this step for now.
    # bias_correct()

Parallel(n_jobs=1)(
    delayed(run_dmri_preproc)(subject_session)
    for subject_session in subjects_sessions)
