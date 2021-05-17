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
5. Bias correction (FAST, FSL)

Currently draft version with hard coded paths etc.
Author: Bertrand Thirion, 2015
Modified by: Swetha Shankar, 2021
"""

import os
import glob
from joblib import Memory, Parallel, delayed
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from ibc_public.utils_pipeline import fsl_topup
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from mayavi import mlab
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

def eddy_current_correction(img, file_bvals, file_bvecs, write_dir, mem,
                            acqp='b0_acquisition_params_AP.txt'):
    """ Perform Eddy current correction on diffusion data
    Todo: do topup + eddy in one single command
    """
    import nibabel as nib
    bvals = np.loadtxt(file_bvals)
    mean_img = get_mean_unweighted_image(nib.load(img), bvals)
    mean_img.to_filename(os.path.join(write_dir, 'mean_unweighted.nii.gz'))
    mask = compute_epi_mask(mean_img)
    mask_file = os.path.join(write_dir, 'mask.nii.gz')
    nib.save(mask, mask_file)
    corrected = os.path.join(os.path.dirname(img),
                             'ed' + os.path.basename(img))
    index = np.ones(len(bvals), np.uint8)
    index_file = os.path.join(write_dir, 'index.txt')
    np.savetxt(index_file, index)
    cmd = 'fsl5.0-eddy_correct --acqp=%s --bvals=%s --bvecs=%s --imain=%s '\
          '--index=%s --mask=%s --out=%s' % (
           acqp, file_bvals, file_bvecs, img, index_file, mask_file, corrected)
    cmd = 'fsl5.0-eddy_correct %s %s %d' % (img, corrected, 0)
    print(cmd)
    os.system(cmd)
    return nib.load(corrected)


def length(streamline):
    """ Compute the length of streamlines"""
    n = streamline.shape[0] // 2
    return np.sqrt((
        (streamline[0] - streamline[n]) ** 2 +
        (streamline[-1] - streamline[n]) ** 2).sum())


def filter_according_to_length(streamlines, threshold=30):
    """Remove streamlines shorter than the predefined threshold """
    print(len(streamlines))
    for i in range(len(streamlines) - 1, 0, -1):
        if length(streamlines[i]) < threshold:
            streamlines.pop(i)

    print(len(streamlines))
    return streamlines


def adapt_ini_file(template, subject, session):
    """ Adapt an ini file by changing the subject and session"""
    output_name = os.path.join(
        '/tmp', os.path.basename(template)[:- 4] + '_' + subject + '_'
        + session + '.ini')
    f1 = open(template, 'r')
    f2 = open(output_name, 'w')
    for line in f1.readlines():
        f2.write(line.replace('sub-01', subject).replace('ses-01', session))

    f1.close()
    f2.close()
    return output_name


def get_mean_unweighted_image(img, bvals):
    """ Create an average diffusion image from the most weakly weighted images
    for registration"""
    X = img.get_data().T[bvals < 50].T
    return nib.Nifti1Image(X.mean(-1), img.affine)


def visualization(streamlines_file):
    # clustering of fibers into bundles and visualization thereof
    streamlines = np.load(streamlines_file)['arr_0']
    qb = QuickBundles(streamlines, dist_thr=10., pts=18)
    centroids = qb.centroids
    colors = line_colors(centroids).astype(np.float)
    mlab.figure(bgcolor=(0., 0., 0.))
    for streamline, color in zip(centroids, colors):
        mlab.plot3d(streamline.T[0], streamline.T[1], streamline.T[2],
                    line_width=1., tube_radius=.5, color=tuple(color))

    figname = streamlines_file[:-3] + 'png'
    mlab.savefig(figname)
    print(figname)
    mlab.close()


def tractography(img, gtab, mask, dwi_dir, do_viz=True):
    data = img.get_data()
    # dirty imputation
    data[np.isnan(data)] = 0
    # Diffusion model
    csd_model = ConstrainedSphericalDeconvModel(gtab, response=None)

    sphere = get_sphere('symmetric724')
    csd_peaks = peaks_from_model(
        model=csd_model, data=data, sphere=sphere, mask=mask,
        relative_peak_threshold=.5, min_separation_angle=25,
        parallel=False)

    # FA values to stop the tractography
    tensor_model = TensorModel(gtab, fit_method='WLS')
    tensor_fit = tensor_model.fit(data, mask)
    fa = fractional_anisotropy(tensor_fit.evals)
    stopping_values = np.zeros(csd_peaks.peak_values.shape)
    stopping_values[:] = fa[..., None]

    # tractography
    streamline_generator = EuDX(stopping_values,
                                csd_peaks.peak_indices,
                                seeds=10**6,
                                odf_vertices=sphere.vertices,
                                a_low=0.1)

    streamlines = [streamline for streamline in streamline_generator]
    streamlines = filter_according_to_length(streamlines)
    np.savez(os.path.join(dwi_dir, 'streamlines.npz'), streamlines)

    #  write the result as images
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.header.get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = fa.shape[:3]

    csd_streamlines_trk = ((sl, None, None) for sl in streamlines)
    csd_sl_fname = os.path.join(dwi_dir, 'csd_streamline.trk')
    nib.trackvis.write(csd_sl_fname, csd_streamlines_trk, hdr,
                       points_space='voxel')
    fa_image = os.path.join(dwi_dir, 'fa_map.nii.gz')
    nib.save(nib.Nifti1Image(fa, img.affine), fa_image)
    if 1:
        visualization(os.path.join(dwi_dir, 'streamlines.npz'))

    return streamlines


def run_dmri_pipeline(subject_session, do_topup=True, do_edc=True):
    subject, session = subject_session
    anat_dir = os.path.join(source_dir, subject, session, 'anat')
    data_dir = os.path.join(source_dir, subject, session, 'dwi')
    fmap_dir = os.path.join(source_dir, subject, session, 'fmap')
    write_dir = os.path.join(derivatives_dir, subject, session)
    dwi_dir = os.path.join(write_dir, 'dwi')
    outfmap_dir = os.path.join(write_dir, 'fmap')

    # Extract T1w brain
    t1_img = glob.glob('%s/sub*T1w.nii.gz' % anat_dir)[0]
    # extracted = extract_brain(t1_img, write_dir, subject, session)

    # Concatenate images
    dc_imgs = sorted(glob.glob(os.path.join(data_dir, '*run*dwi.nii.gz')))
    # nib.nifti1.save(nib.funcs.concat_images(dc_imgs, axis=3), dc_img)

    # Concatenate the bval and bvec files as well
    # bval_files = sorted(glob.glob(os.path.join(data_dir, '*run*dwi.bval')))
    bvals_out = os.path.join(dwi_dir, "bvals")
    # bvals_ = np.loadtxt(bval_files[0], dtype=int)
    # for i in range(len(bval_files)-1):
    #     bv = np.loadtxt(bval_files[i+1], dtype=int)
    #     bvals_ = np.concatenate((bvals_,bv))
    # np.savetxt(bvals_out, bvals_, fmt='%d', newline=' ')

    # bvec_files = sorted(glob.glob(os.path.join(data_dir, '*run*dwi.bvec')))
    bvecs_out = os.path.join(dwi_dir, "bvecs")
    # bvecs_ = np.loadtxt(bvec_files[0])
    # for i in range(len(bvec_files)-1):
    #     bv = np.loadtxt(bvec_files[i+1])
    #     bvecs_ = np.concatenate((bvecs_,bv), axis=1)
    # np.savetxt(bvecs_out, bvecs_)


    # Denoise images using MP-PCA
    dn_img = os.path.join(dwi_dir, '%s_%s_dwi-denoise.nii.gz' % (subject, session))
    # cmd = 'dwidenoise %s %s' % (dc_img, dn_img)
    # print(cmd)
    # os.system(cmd)

    # Remove Gibbs ringing artifacts
    # The recommendation is to do this if not using partial Fourier acquisition, but we do use it.
    # The images look a little blurred and of lower intensity than the denoised images
    dg_img = os.path.join(dwi_dir, '%s_%s_dwi-degibbs.nii.gz' % (subject, session))
    # cmd = 'mrdegibbs %s %s' % (dn_img, dg_img)
    # print(cmd)
    # os.system(cmd)

    # Run FSL topup
    fmap_imgs = sorted(glob.glob('%s/sub*dir*epi.nii.gz' % fmap_dir))
    merged_img = os.path.join(outfmap_dir, '%s_%s_dir-appa_epi.nii.gz' % (subject, session))
    # if len(fmap_imgs)==2:
    #     cmd = "fslmerge -t %s %s %s" % (
    #         merged_img, fmap_imgs[0], fmap_imgs[1])
    #     print(cmd)
    #     os.system(cmd)
    # else:
    #     print("There are more than 2 SE files, can't proceed")

    acq_params_file = os.path.join(fmap_dir, 'b0_acquisition_params.txt')
    topup_results_basename = os.path.join(outfmap_dir, 'topup_result')
    hifi_file = os.path.join(outfmap_dir, 'hifi_b0')
    # cmd = "topup --imain=%s --datain=%s --config=b02b0.cnf --out=%s --iout=%s" % (
    #     merged_img, acq_params_file, topup_results_basename, hifi_file)
    # print(cmd)
    # os.system(cmd)

    # Calculate the mean image of the hifi_file and etract the brain from it before running eddy
    # cmd = "fslmaths %s -Tmean %s" % (hifi_file, hifi_file)
    # print(cmd)
    # os.system(cmd)

    hifi_file_brain = os.path.join(outfmap_dir, 'hifi_b0_brain')
    # cmd = "bet %s %s -f 0.85 -R -m" % (hifi_file, hifi_file_brain)
    # print(cmd)
    # os.system(cmd)

    # Create a text file that contains, for each volume in the concatenated dwi images file,
    # the corresponding line of the acquisitions parameter file. The way the data has been
    # concatenated, we have 2 AP runs followed by 2 PA runs, each with 61 volumes. Thus, the
    # text file will have 244 lines, the first 122 will say "1" and the last 122 will say "2"
    index_file = os.path.join(dwi_dir, 'index.txt')
    # inds = np.concatenate((np.ones(122, dtype=int), np.ones(122, dtype=int)*2))
    # np.savetxt(index_file, inds, fmt='%d')

    # Now run eddy to correct eddy current distortions
    mask_img = glob.glob('%s/*mask.nii.gz' % outfmap_dir)[0]

    # eddy_out = os.path.join(dwi_dir, 'eddy_denoise')
    # # This command uses the denoised images as input
    # cmd = "eddy --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --repol --out=%s" % (
    #     dn_img, mask_img, acq_params_file, index_file, bvecs_out, bvals_out, topup_results_basename, eddy_out)

    eddy_out = os.path.join(dwi_dir, 'eddy_degibbs')
    # This command uses the degibbsed images as input
    cmd = "eddy --imain=%s --mask=%s --acqp=%s --index=%s --bvecs=%s --bvals=%s --topup=%s --repol --out=%s" % (
        dg_img, mask_img, acq_params_file, index_file, bvecs_out, bvals_out, topup_results_basename, eddy_out)

    print(cmd)
    os.system(cmd)

    # # Apply topup to the images
    # input_imgs = sorted(glob.glob('%s/sub*.nii.gz' % data_dir))
    # dc_imgs = sorted(glob.glob(os.path.join(dwi_dir, 'dcsub*run*.nii.gz')))
    # # mem = Memory('/neurospin/tmp/bthirion/cache_dir')
    # if len(dc_imgs) < len(input_imgs):
    #     se_maps = [
    #         os.path.join(fmap_dir, '%s_%s_dir-ap_epi.nii.gz' % (subject, session)),
    #         os.path.join(fmap_dir, '%s_%s_dir-pa_epi.nii.gz' % (subject, session))]
    #
    #     if do_topup:
    #         fsl_topup(se_maps, input_imgs, mem, write_dir, 'dwi')
    #
    # # Then proceeed with Eddy current correction
    # # get the images
    # dc_imgs = sorted(glob.glob(os.path.join(dwi_dir, 'dc*run*.nii.gz')))
    # dc_img = os.path.join(dwi_dir, 'dc%s_%s_dwi.nii.gz' % (subject, session))
    # concat_images(dc_imgs, dc_img)
    #
    # # get the bvals/bvec
    # file_bvals = sorted(glob.glob('%s/sub*.bval' % data_dir))
    # bvals = np.concatenate([np.loadtxt(fbval) for fbval in sorted(file_bvals)])
    # bvals_file = os.path.join(dwi_dir, 'dc%s_%s_dwi.bval' % (subject, session))
    # np.savetxt(bvals_file, bvals)
    # file_bvecs = sorted(glob.glob('%s/sub*.bvec' % data_dir))
    # bvecs = np.hstack([np.loadtxt(fbvec) for fbvec in sorted(file_bvecs)])
    # bvecs_file = os.path.join(dwi_dir, 'dc%s_%s_dwi.bvec' % (subject, session))
    # np.savetxt(bvecs_file, bvecs)
    #
    # # Get eddy-preprocessed images
    # # eddy_img = nib.load(glob.glob(os.path.join(dwi_dir, 'eddc*.nii*'))[-1])
    #
    # # Get eddy-preprocessed images
    # eddy_img = mem.cache(eddy_current_correction)(
    #     dc_img, bvals_file, bvecs_file, dwi_dir, mem)
    #
    # # load the data
    # gtab = gradient_table(bvals, bvecs, b0_threshold=10)
    # # Create a brain mask
    #
    # from dipy.segment.mask import median_otsu
    # b0_mask, mask = median_otsu(eddy_img.get_data(), 2, 1)
    # if subject == 'sub-13':
    #     from nilearn.masking import compute_epi_mask
    #     from nilearn.image import index_img
    #     imgs_ = [index_img(eddy_img, i)
    #              for i in range(len(bvals)) if bvals[i] < 50]
    #     mask_img = compute_epi_mask(imgs_, upper_cutoff=.8)
    #     mask_img.to_filename('/tmp/mask.nii.gz')
    #     mask = mask_img.get_data()
    # # do the tractography
    # streamlines = tractography(eddy_img, gtab, mask, dwi_dir)
    # return streamlines
    #

Parallel(n_jobs=1)(
    delayed(run_dmri_pipeline)(subject_session, do_topup, do_edc)
    for subject_session in subjects_sessions)

# mlab.show()
