"""
Steps for the pre-processing of dMRI data as recommended by Ades-Aron, B et al, 2019, Neuroimage
1. Noise correction (MP_PCA)
2. Gibbs ringing correction
3. Rician correction
4. Distortion correction
5. Eddy current and motion correction
6. B1 inhomogeneity correction
7. Outlier detection

Diffusion processing steps as implemented by the NSD project (http://naturalscenesdataset.org/),
following recommendations from the above paper:
1. Gradient nonlinearity correction
2. Concatenation of all runs
3. Denoising
4. Gibbs ringing correction (mrdegibbs, MRTrix3)
5. Distortion correction (topup, FSL)
6. Eddy correction (eddy, FSL)
7. Bias correction and noise removal (MRTrix3)
With anatomical:
1. Segmentation of T1w
2. Registration of diffusion volumes to T1w image (epi-reg, FSL)
3. Resampling to 0.8 mm isotropic voxels
7. Assess data quality by calculating SNR in corpus callosum (dipy)
8. Generate brain mask (median_otsu, dipy)
9. For network generation, the HCP-MMP cortical parcellation (Glasser et al., 2016) was mapped to subject-native
   surfaces and then to the volumetric Freesurfer segmentation (ribbon.mgz) for each subject

Maximov, II et al, 2019, Human Brain Mapping (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.24691)
compare diffusion processing steps and recommend the following steps at a minimum:
1. Noise correction (MP_PCA)
2. Gibbs ringing correction (MRTrix3)
3. Distortion correction (topup, FSL)
4. Eddy correction (eddy, FSL)
5. Bias field correction (ANTs or FAST, FSL)
6. Spatial smoothing (fslmaths, FSL)
7. Metric estimations

To implement:
0. Concatenate runs
1. Noise correction (dwidenoise, MRTrix3 - uses MP_PCA algorithm)
2. Gibbs ringing correction (mrdegibbs, MRTrix3)
    According to mrtrix documentation, running this method on partial Fourier (‘half-scan’) data may lead
    to suboptimal and/or biased results, so exert caution and look at your results. We use partial Fourier (6/8)
    in our diffusion acquisition
3. Distortion correction (topup, FSL)
    One can feed the output from topup into the eddy tool
4. Eddy current and motion correction (eddy, FSL)
5. Bias correction (dwibiascorrect, ANTS via MRTrix3 - not working well, skipping this step)
6. Register DWI images to each other and then to T1 (flirt and epi_reg, FSL)

Currently draft version with hard coded paths etc.
Author: Bertrand Thirion, 2015
Modified by: Swetha Shankar, 2021
"""

import os
import glob
from joblib import Memory, Parallel, delayed
import numpy as np
import nibabel as nib
# from ibc_public.utils_pipeline import fsl_topup
from dipy.segment.mask import median_otsu
from dipy.align import register_dwi_series, register_dwi_to_template
import dipy.reconst.dti as dti
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import recursive_response, ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.data import get_sphere, default_sphere
from dipy.core.gradients import gradient_table
# from mayavi import mlab
from ibc_public.utils_data import get_subject_session


source_dir = '/home/sshankar/diffusion/sourcedata'
derivatives_dir = '/home/sshankar/diffusion/derivatives'
do_topup = False
do_edc = 0
subjects_sessions = [('sub-04', 'ses-08')]  # get_subject_session('anat1')

def extract_brain(t1_img, out_dir, subject, session):
    # I tried median_otsu for brain extraction but while it did do brain extaction, it
    # did not do skull stripping. This made file size ~3 times larger than what bet gives.
    t1_brain = os.path.join(out_dir, 'anat', '%s_%s_t1w_bet.nii.gz' % (subject, session))
    cmd = 'bet %s %s -m' % (t1_img, t1_brain)
    print(cmd)
    os.system(cmd)

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

def align_t1_dwi(b0_vol, hires_t1, t1_aligned):
    cmd = "flirt --in=%s --ref=%s --out=%s" % (
        hires_t1, b0_vol, t1_aligned)

    print(cmd)
    os.system(cmd)

def tractography(img, gtab, mask, dwi_dir, do_viz=True):
    data = img.get_fdata()
    # dirty imputation
    data[np.isnan(data)] = 0

    # Estimate fiber response function by using a data-driven calibration strategy
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=data[..., 0] > 200)

    FA = fractional_anisotropy(tenfit.evals)
    MD = dti.mean_diffusivity(tenfit.evals)
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

    response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                                  peak_thr=0.01, init_fa=0.08,
                                  init_trace=0.0021, iter=8, convergence=0.001,
                                  parallel=True)

    response_signal = response.on_sphere(default_sphere)
    # transform our data from 1D to 4D
    response_signal = response_signal[None, None, None, :]
    response_actor = actor.odf_slicer(response_signal, sphere=default_sphere,
                                      colormap='plasma')

    scene = window.Scene()

    scene.add(response_actor)
    print('Saving illustration as csd_recursive_response.png')
    window.record(scene, out_path='csd_recursive_response.png', size=(200, 200))
    if interactive:
        window.show(scene)
    scene.rm(response_actor)

    # # Diffusion model
    # csd_model = ConstrainedSphericalDeconvModel(gtab)
    #
    # sphere = get_sphere('symmetric724')
    # csd_peaks = peaks_from_model(
    #     model=csd_model, data=data, sphere=sphere, mask=mask,
    #     relative_peak_threshold=.5, min_separation_angle=25,
    #     parallel=False)
    #
    # # FA values to stop the tractography
    # tensor_model = TensorModel(gtab, fit_method='WLS')
    # tensor_fit = tensor_model.fit(data, mask)
    # fa = fractional_anisotropy(tensor_fit.evals)
    # stopping_values = np.zeros(csd_peaks.peak_values.shape)
    # stopping_values[:] = fa[..., None]
    #
    # # tractography
    # streamline_generator = EuDX(stopping_values,
    #                             csd_peaks.peak_indices,
    #                             seeds=10**6,
    #                             odf_vertices=sphere.vertices,
    #                             a_low=0.1)
    #
    # streamlines = [streamline for streamline in streamline_generator]
    # streamlines = filter_according_to_length(streamlines)
    # np.savez(os.path.join(dwi_dir, 'streamlines.npz'), streamlines)
    #
    # #  write the result as images
    # hdr = nib.trackvis.empty_header()
    # hdr['voxel_size'] = img.header.get_zooms()[:3]
    # hdr['voxel_order'] = 'LAS'
    # hdr['dim'] = fa.shape[:3]
    #
    # csd_streamlines_trk = ((sl, None, None) for sl in streamlines)
    # csd_sl_fname = os.path.join(dwi_dir, 'csd_streamline.trk')
    # nib.trackvis.write(csd_sl_fname, csd_streamlines_trk, hdr,
    #                    points_space='voxel')
    # fa_image = os.path.join(dwi_dir, 'fa_map.nii.gz')
    # nib.save(nib.Nifti1Image(fa, img.affine), fa_image)
    # if 1:
    #     visualization(os.path.join(dwi_dir, 'streamlines.npz'))
    #
    # return streamlines

def run_dmri_pipeline(subject_session, do_topup=True, do_edc=True):
    subject, session = subject_session

    src_dir = os.path.join(source_dir, subject, session)
    dest_dir = os.path.join(derivatives_dir, subject, session)

    src_anat_dir = os.path.join(src_dir, 'anat')
    src_dwi_dir = os.path.join(src_dir, 'dwi')
    src_fmap_dir = os.path.join(src_dir, 'fmap')

    dest_dwi_dir = os.path.join(dest_dir, 'dwi')
    dest_fmap_dir = os.path.join(dest_dir, 'fmap')
    dest_anat_dir = os.path.join(dest_dir, 'anat')

    # Extract T1w brain
    t1_img = glob.glob('%s/sub*T1w.nii.gz' % src_anat_dir)[0]
    # extracted = extract_brain(t1_img, write_dir, subject, session)

    # Concatenate images
    dwi_imgs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.nii.gz')))
    out_concat = os.path.join(dest_dwi_dir, '%s_%s_dwi.nii.gz' % (subject, session))
    # concat_images(dwi_imgs, out_concat)

    # Concatenate the bval and bvec files as well
    in_bvals = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bval')))
    out_bvals = os.path.join(dest_dwi_dir, "bvals")
    # concat_bvals(in_bvals, out_bvals)

    in_bvecs = sorted(glob.glob(os.path.join(src_dwi_dir, '*run*dwi.bvec')))
    out_bvecs = os.path.join(dest_dwi_dir, "bvecs")
    # concat_bvecs(in_bvecs, out_bvecs)

    # Denoise images using MP-PCA
    out_dn = os.path.join(dest_dwi_dir, 'dn_%s_%s_dwi.nii.gz' % (subject, session))
    # denoise_dwi(out_concat, out_dn)

    # Remove Gibbs ringing artifacts
    # The recommendation is to do this if not using partial Fourier acquisition, but we do use it.
    # The images look a little blurred and of lower intensity than the denoised images
    out_dg = os.path.join(dest_dwi_dir, 'dg_%s_%s_dwi.nii.gz' % (subject, session))
    # degibbs_dwi(out_dn, out_dg)

    # Run FSL topup - it's a 2-step process
    # 1. Collect all the b=0 volumes in one file and use that as input to topup
    b0_imgs = sorted(glob.glob('%s/dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session)))[0]
    merged_b0_img = os.path.join(dest_dwi_dir, 'b0s_%s_%s_dwi.nii.gz' % (subject, session))
    vols = [0, 61, 122, 183]
    # collate_b0s(b0_imgs, vols, merged_b0_img)

    # 2. Calculate distortion from the collated b0 images
    acq_params_file = os.path.join(src_dwi_dir, 'b0_acquisition_params.txt')
    topup_results_basename = os.path.join(dest_dwi_dir, '%s_%s_topup-results' % (subject, session))
    hifi_file = os.path.join(dest_dwi_dir, '%s_%s_hifi-b0' % (subject, session))
    # calc_topup(merged_b0_img, acq_params_file, hifi_file, topup_results_basename)

    # Calculate the mean image of the hifi_file and extract the brain from it before running eddy
    hifi_brain = os.path.join(dest_dwi_dir, '%s_%s_hifi-b0-brain.nii.gz')
    threshold = 0.67
    # make_hifi_mask(hifi_file, threshold, hifi_brain)

    # Create a text file that contains, for each volume in the concatenated dwi images file,
    # the corresponding line of the acquisitions create_parameter file.
    # The way the data has been concatenated, we have 2 AP runs followed by 2 PA runs,
    # each with 61 volumes. Thus, the text file will have 244 lines, the first 122 will
    # say "1" and the last 122 will say "2"
    index_file = os.path.join(dest_dwi_dir, 'dwi_acqdir_index.txt')
    # inds = np.concatenate((np.ones(122, dtype=int), np.ones(122, dtype=int)*2))
    # np.savetxt(index_file, inds, fmt='%d')

    # Now run eddy to correct eddy current distortions
    mask_img = glob.glob('%s/*mask.nii.gz' % dest_dwi_dir)[0]
    eddy_in = out_dn
    eddy_out = os.path.join(dest_dwi_dir, 'eddy_dn_%s_%s_dwi.nii.gz' % (subject, session))
    # run_eddy(eddy_in, mask_img, acq_params_file, index_file, out_bvecs, out_bvals, topup_results_basename, eddy_out)

    # Once again extract the b0 volumes, this time from the eddy corrected images,
    # create a mean volume, and a mask of the mean volume
    b0_imgs = glob.glob('%s/eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    print(b0_imgs)
    merged_b0_img = os.path.join(dest_dwi_dir, 'b0s_eddy_dn_%s_%s_dwi.nii.gz' % (subject, session))
    vols = [0, 61, 122, 183]
    # collate_b0s(b0_imgs, vols, merged_b0_img)
    b0_brain = os.path.join(dest_dwi_dir, 'b0_brain_eddy_dn_%s_%s_dwi' % (subject, session))
    # make_hifi_mask(merged_b0_img, threshold, b0_brain)

    # Bias field correction
    # Bias field correction doesn't work very well via dwibiascorrect, and
    # fails when I try running N4BiasFieldCorrection. Skipping this step for now.
    # bias_correct()

    # Align T1w volume to DWI volumes
    hires_t1 = glob.glob('%s/*T1w*' % src_anat_dir)[0]
    t1_aligned = os.path.join(dest_dwi_dir, 'dwi-aligned-T1_%s_%s' %(subject, session))
    # align_t1_dwi(hires_t1, b0_vol, t1_aligned)

    # load the data
    gtab = gradient_table(out_bvals, out_bvecs, b0_threshold=10)

    # do the tractography
    b0_mask = glob.glob('%s/b0_brain_eddy_dn_%s_%s_dwi_mask.nii.gz' % (dest_dwi_dir, subject, session))[0]

    tractography(nib.load(eddy_out), gtab, b0_mask, dest_dwi_dir)

    # streamlines = tractography(nib.load(eddy_out), gtab, b0_mask, dest_dwi_dir)
    # return streamlines

Parallel(n_jobs=1)(
    delayed(run_dmri_pipeline)(subject_session, do_topup, do_edc)
    for subject_session in subjects_sessions)

# mlab.show()
