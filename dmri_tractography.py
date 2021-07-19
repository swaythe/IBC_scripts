import os
import glob
from joblib import Memory, Parallel, delayed
import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import recursive_response, ConstrainedSphericalDeconvModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import peaks_from_model
from dipy.data import get_sphere, default_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk, load_trk
from dipy.segment.clustering import QuickBundles
from ibc_public.utils_data import get_subject_session

derivatives_dir = '/home/sshankar/diffusion/derivatives'
subject = 'sub-04'
session = 'ses-08'
dest_dir = os.path.join(derivatives_dir, subject, session)
dest_dwi_dir = os.path.join(dest_dir, 'dwi')

b0_mask = glob.glob('%s/b0_brain_eddy_dn_%s_%s_dwi_mask.nii.gz' % (dest_dwi_dir, subject, session))[0]
mask = nib.load(b0_mask)

seeds = random_seeds_from_mask(mask=b0_mask_img, affine=b0_mask_img.affine, seeds_count=10**6)
fa_file = os.path.join(dest_dwi_dir, "fa_map.nii.gz")
wm_file = os.path.join(dest_dwi_dir, "wm_mask.nii.gz")
trk_file = os.path.join(dest_dwi_dir, "tractogram_EuDX.trk")

def tractography(img, gtab, mask, seeds, dwi_dir, fa_file, wm_file, trk_file):
eddy_out = glob.glob('%s/eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
img = nib.load(eddy_out)
data = img.get_fdata()

out_bvals = glob.glob('%s/bvals' % dest_dwi_dir)[0]
out_bvecs = glob.glob('%s/bvecs' % dest_dwi_dir)[0]
gtab = gradient_table(out_bvals, out_bvecs, b0_threshold=10)

mask_data = mask.get_fdata()
# dirty imputation
data[np.isnan(data)] = 0

sphere = get_sphere('symmetric724')

# Estimate fiber response function by using a data-driven calibration strategy
tensor_model = dti.TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask=mask_data)

FA = fractional_anisotropy(tensor_fit.evals)
nib.save(nib.Nifti1Image(FA, img.affine), fa_file)

MD = dti.mean_diffusivity(tensor_fit.evals)
wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
# nib.save(nib.Nifti1Image(wm_mask, img.affine, img.header), wm_file)

response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True)

# response_signal = response.on_sphere(sphere)
# # Transform data from 1D to 4D
# response_signal = response_signal[None, None, None, :]
# response_actor = actor.odf_slicer(response_signal, sphere=sphere)
#
# scene = window.Scene()
#
# scene.add(response_actor)
# print('Saving illustration as csd_recursive_response.png')
# window.record(scene, out_path=os.path.join(dwi_dir,'csd_recursive_response.png'), size=(200, 200))
# scene.rm(response_actor)

# Deconvolution
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data)

# Show the CSD-based orientation distribution functions (ODFs) also known as FODFs (fiber ODFs)
csd_odf = csd_fit.odf(sphere)
# Visualize a small (30x30) region
# fodf_spheres = actor.odf_slicer(csd_odf, sphere=sphere, scale=0.9, norm=False)
# scene.add(fodf_spheres)
# print('Saving illustration as csd_odfs.png')
# window.record(scene, out_path=os.path.join(dwi_dir,'csd_odfs.png'), size=(600, 600))

# Find the peak directions (maxima) of the ODFs
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=mask_data,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

# Some visualizations
# scene.clear()
# fodf_peaks = actor.peak_slicer(csd_peaks.peak_dirs, csd_peaks.peak_values)
# scene.add(fodf_peaks)
# print('Saving illustration as csd_peaks.png')
# window.record(scene, out_path=os.path.join(dwi_dir,'csd_peaks.png'), size=(600, 600))
#
# # Visualize both the ODFs and peaks in the same space
# fodf_spheres.GetProperty().SetOpacity(0.4)
# scene.add(fodf_spheres)
# print('Saving illustration as csd_both.png')
# window.record(scene, out_path=os.path.join(dwi_dir,'csd_both.png'), size=(600, 600))

# if has_fury:
#     scene = window.Scene()
#     scene.add(actor.peak_slicer(csd_peaks.peak_dirs,
#                                 csd_peaks.peak_values,
#                                 colors=None))
#
#     window.record(scene, out_path=os.path.join(dwi_dir,'csd_direction_field.png'), size=(900, 900))

# tractography
stopping_criterion = ThresholdStoppingCriterion(csd_peaks.gfa, .25)
print(stopping_criterion)
print(csd_peaks.gfa.shape)

# sli = csd_peaks.gfa.shape[2]
# plt.figure('GFA')
# plt.subplot(1, 2, 1).set_axis_off()
# plt.imshow(csd_peaks.gfa[:, :, sli].T, cmap='gray', origin='lower')
# plt.subplot(1, 2, 2).set_axis_off()
# plt.imshow((csd_peaks.gfa[:, :, sli] > 0.25).T, cmap='gray', origin='lower')
# plt.savefig(os.path.join(dwi_dir,'gfa_tracking_mask.png'))

# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csd_peaks, stopping_criterion, seeds=seeds,
                                      affine=img.affine, step_size=.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)
np.savez(os.path.join(dwi_dir, 'streamlines.npz'), streamlines)

# Display resulting streamlines
# if has_fury:
#     # Prepare the display objects.
#     streamlines_actor = actor.line(streamlines, colormap.line_colors(streamlines))
#
#     # Create the 3D display.
#     scene = window.Scene()
#     scene.add(streamlines_actor)
#
#     # Save still images for this static example. Or for interactivity use
#     window.record(scene, out_path=os.path.join(dwi_dir,'tractogram_EuDX.png'), size=(800, 800))
#     # if interactive:
#     #     window.show(scene)

sft = StatefulTractogram(streamlines, img, Space.RASMM)
save_trk(sft, trk_file, streamlines)
