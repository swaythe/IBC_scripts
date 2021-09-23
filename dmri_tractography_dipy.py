import os
import glob
from joblib import Memory, Parallel, delayed
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity
from dipy.reconst.csdeconv import recursive_response, ConstrainedSphericalDeconvModel
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import random_seeds_from_mask
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.segment.clustering import QuickBundlesX
from mayavi import mlab
from dipy.viz import colormap

derivatives_dir = '/home/sshankar/diffusion/derivatives'
subject = 'sub-04'
session = 'ses-08'
dest_dir = os.path.join(derivatives_dir, subject, session)
dest_dwi_dir = os.path.join(dest_dir, 'dwi')

b0_mask = glob.glob('%s/b0_brain_eddy_dn_%s_%s_dwi_mask.nii.gz' % (dest_dwi_dir, subject, session))[0]
mask = nib.load(b0_mask)

eddy_out = glob.glob('%s/eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
img = nib.load(eddy_out)
data = img.get_fdata()

out_bvals = glob.glob('%s/bvals' % dest_dwi_dir)[0]
out_bvecs = glob.glob('%s/bvecs' % dest_dwi_dir)[0]
gtab = gradient_table(out_bvals, out_bvecs, b0_threshold=10)

mask_data = mask.get_fdata()
# dirty imputation
# data[np.isnan(data)] = 0

sphere = get_sphere('symmetric724')

# Estimate fiber response function by using a data-driven calibration strategy
tensor_model = TensorModel(gtab)
tensor_fit = tensor_model.fit(data, mask=mask_data)

FA = fractional_anisotropy(tensor_fit.evals)
fa_file = os.path.join(dest_dwi_dir, "fa_map.nii.gz")
nib.save(nib.Nifti1Image(FA, img.affine), fa_file)

MD = mean_diffusivity(tensor_fit.evals)
# wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
wm_mask = (np.logical_or(FA >= 0.5, (np.logical_and(FA >= 0.15, MD >= 0.0015))))
wm_file = os.path.join(dest_dwi_dir, "wm_mask_fa5.nii.gz")
nib.save(nib.Nifti1Image(wm_mask, img.affine, img.header), wm_file)

# seeds = random_seeds_from_mask(mask=mask, affine=mask.affine, seeds_count=10**6)
seeds = random_seeds_from_mask(mask=wm_mask, affine=mask.affine, seeds_count=10**2)
response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                              peak_thr=0.01, init_fa=0.08,
                              init_trace=0.0021, iter=8, convergence=0.001,
                              parallel=True, sphere=sphere)
np.savez(os.path.join(dest_dwi_dir, 'recursive_response.npz'), response)

# Deconvolution
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

# Find the peak directions (maxima) of the ODFs
csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=wm_mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)

# tractography
stopping_criterion = ThresholdStoppingCriterion(csd_peaks.gfa, .2)

# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csd_peaks, stopping_criterion, seeds=seeds,
                                      affine=img.affine, step_size=.5)
# Generate streamlines object
streamlines = Streamlines(streamlines_generator)
np.savez(os.path.join(dest_dwi_dir, 'streamlines.npz'), streamlines)

sft = StatefulTractogram(streamlines, img, Space.RASMM)
trk_file = os.path.join(dest_dwi_dir, "tractogram_EuDX.trk")
save_trk(sft, trk_file, streamlines)

# Clustering the tractogram
thresholds = [10.,15.,20.]
qb = QuickBundlesX(thresholds)
clusters = qb.cluster(streamlines)

tree = clusters.get_clusters(len(thresholds))
tree.refdata = streamlines
centroids = tree.centroids
np.savez(os.path.join(dest_dwi_dir, 'centroids.npz'), centroids)

# Plot the streamlines
colors = colormap.line_colors(centroids).astype(float)

mlab.figure(bgcolor=(0., 0., 0.))

for streamline, color in zip(centroids, colors):
    mlab.plot3d(streamline.T[0], streamline.T[1], streamline.T[2],
                line_width=1., tube_radius=.5, color=tuple(color))

figname = 'streamlines.png'
mlab.savefig(os.path.join(dest_dwi_dir, figname))
mlab.close()
