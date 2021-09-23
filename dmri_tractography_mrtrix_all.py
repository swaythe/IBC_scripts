import os
import glob
import matplotlib.pyplot as plt
import numpy as np

source_dir = '/home/sshankar/diffusion/sourcedata'
derivatives_dir = '/home/sshankar/diffusion/derivatives'
subject = 'sub-04'
session = 'ses-08'
src_dir = os.path.join(source_dir, subject, session)
src_anat_dir = os.path.join(src_dir, 'anat')
dest_dir = os.path.join(derivatives_dir, subject, session)
dest_dwi_dir = os.path.join(dest_dir, 'dwi')
dest_anat_dir = os.path.join(dest_dir, 'anat')

def convert_mif(in_file, out_file):
    cmd = 'mrconvert %s %s' % (in_file, out_file)
    print(cmd)
    os.system(cmd)

def calc_basis_fn(algorithm='dhollander', eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out):
    cmd = 'dwi2response %s %s %s %s %s -fslgrad %s %s -mask %s -voxels %s' % (algorithm, eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out)
    print(cmd)
    os.system(cmd)

def create_fod(eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod):
    # Uses multi-shell multi-tissue constrained spherical deconvolution
    cmd = 'dwi2fod msmt_csd %s -fslgrad %s %s -mask %s -force %s %s %s %s %s %s' % (eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod)
    print(cmd)
    os.system(cmd)

def seg_anat(algorithm='fsl', t1_mif, seg_out):
    cmd = '5ttgen %s %s %s' % (algorithm, t1_mif, seg_out)
    print(cmd)
    os.system(cmd)

def align_dwi_t1(mean_b0, t1_nifti, transform_mat):
    cmd = 'flirt -in %s -ref %s -omat %s' % (mean_b0, t1_nifti, transform_mat)
    # print(cmd)
    # os.system(cmd)

def run_dmri_tractography(subject_session):
    # Convert DWI files to mif format
    eddy_in = glob.glob('%s/eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    eddy_mif = os.path.join(dest_dwi_dir, 'eddy_dn_%s_%s_dwi.mif' % (subject, session))
    convert_mif(eddy_in, eddy_mif)

    # Derive basis functions for the different tissue types from diffusion data using the Dhollander algorithm.
    # The wm, gm and csf txt files contain the response functions for those tissue types.
    # These are all generated using the dwi2response function
    algorithm = 'dhollander'
    bvecs = glob.glob('%s/bvecs' % dest_dwi_dir)[0]
    bvals = glob.glob('%s/bvals' % dest_dwi_dir)[0]
    mask = glob.glob('%s/b0_brain_eddy_dn_%s_%s_dwi_mask.nii.gz' % (dest_dwi_dir, subject, session))[0]
    wm_out = os.path.join(dest_dwi_dir, 'wm_%s_%s_dwi.txt' % (subject, session))
    gm_out = os.path.join(dest_dwi_dir, 'gm_%s_%s_dwi.txt' % (subject, session))
    csf_out = os.path.join(dest_dwi_dir, 'csf_%s_%s_dwi.txt' % (subject, session))
    voxels_out = os.path.join(dest_dwi_dir, 'voxels_%s_%s_dwi.mif' % (subject, session))
    calc_basis_fn(algorithm, eddy_mif, wm_out, gm_out, csf_out, bvecs, bvals, mask, voxels_out)
    # Use mrview to visualize the voxels file to make sure voxels are in the correct tissue group.
    # Red markers should be in CSF, Green markers should be in gray matter, Blue markers should be in white matter
    # View the basis functions files using shview.

    # Using the basis functions we can create the FODs, or fiber orientation densities. These are estimates of the amount of diffusion in the 3 orthogonal directions.
    wm_fod = os.path.join(dest_dwi_dir, 'wm-fod_%s_%s_dwi.mif' % (subject, session))
    gm_fod = os.path.join(dest_dwi_dir, 'gm-fod_%s_%s_dwi.mif' % (subject, session))
    csf_fod = os.path.join(dest_dwi_dir, 'csf-fod_%s_%s_dwi.mif' % (subject, session))
    create_fod(eddy_mif, bvecs, bvals, mask, wm_out, wm_fod, gm_out, gm_fod, csf_out, csf_fod)

    # Concatenate the gm and csf fods with the first volume of the wm fod. Not sure why the wm fod file alone has multiple volumes.
    # concat_fod = os.path.join(dest_dwi_dir, 'concat-fod_%s_%s_dwi.mif' % (subject, session))
    # cmd = 'mrconvert -coord 3 0 %s - | mrcat %s %s -force - %s' % (wm_fod, gm_fod, csf_fod, concat_fod)
    # print(cmd)
    # os.system(cmd)
    # To view use the following command:
    # mrview concat-fod*.mif -odf.load_sh wm-fod*.mif
    # The color scheme is:
    # Red = L-R, Green = P-A, Blue = I-S

    # Next step is to normalize the FOD values to be able to compare between subjects
    # wm_fod_norm = os.path.join(dest_dwi_dir, 'wm-fod-norm_%s_%s_dwi.mif' % (subject, session))
    # gm_fod_norm = os.path.join(dest_dwi_dir, 'gm-fod-norm_%s_%s_dwi.mif' % (subject, session))
    # csf_fod_norm = os.path.join(dest_dwi_dir, 'csf-fod-norm_%s_%s_dwi.mif' % (subject, session))
    #
    # cmd = 'mtnormalise -mask %s -force %s %s %s %s %s %s' % (mask, wm_fod, wm_fod_norm, gm_fod, gm_fod_norm, csf_fod, csf_fod_norm)
    # print(cmd)
    # os.system(cmd)

    # Convert the anatomical to MRtrix format
    t1_nifti = glob.glob('%s/*T1w.nii.gz' % (src_anat_dir))[0]
    t1_mif = os.path.join(dest_anat_dir, '%s_%s_acq-highres_T1w.mif' % (subject, session))
    convert_mif(t1_nifti, t1_mif)

    # Segment anatomical into individual tissue types using FSL (other options available)
    seg_out = os.path.join(dest_dwi_dir, 'seg_%s_%s_T1w.mif' % (subject, session))
    algorithm = 'fsl'
    seg_anat(algorithm, t1_mif, seg_out)

    # Align the mean b0 image to the anatomical using FLIRT
    mean_b0 = glob.glob('%s/b0_brain_eddy_dn_%s_%s_dwi.nii.gz' % (dest_dwi_dir, subject, session))[0]
    transform_mat = os.path.join(dest_dwi_dir, '%s_%s_dwi-to-T1.mat' % (subject, session))
    align_dwi_t1(mean_b0, t1_nifti, transform_mat)

# Convert the transformation matrix to MRtrix format
transform_mat = os.path.join(dest_dwi_dir, 'dwi_to_t1.mat')
transform_mat_mrtrix = os.path.join(dest_dwi_dir, '%s_%s_dwi-to-T1.txt' % (subject, session))
cmd = 'transformconvert %s %s %s flirt_import %s' % (transform_mat, mean_b0, t1_nifti, transform_mat_mrtrix)
# print(cmd)
# os.system(cmd)

# Apply the transformation matrix to the non-coregistered segmentation data:
seg_out_coreg = os.path.join(dest_dwi_dir, 'coreg-seg_%s_%s_T1w.mif' % (subject, session))
cmd = 'mrtransform %s -linear %s -inverse %s' % (seg_out, transform_mat_mrtrix, seg_out_coreg)
# print(cmd)
# os.system(cmd)

# Generate GM/WM boundary
gmwm_bound = os.path.join(dest_dwi_dir, 'gmwm-bound-coreg_%s_%s.mif' % (subject, session))
cmd = '5tt2gmwmi %s %s' % (seg_out_coreg, gmwm_bound)
# print(cmd)
# os.system(cmd)

# Create streamlines
track_out = os.path.join(dest_dwi_dir, 'tracks_%s_%s_t1.tck' % (subject, session))
cmd = 'tckgen -act %s -backtrack -seed_gmwmi %s -maxlength 250 -cutoff 0.06 -select 10000000 %s %s' % (seg_out_coreg, gmwm_bound, wm_fod_norm, track_out)
# print(cmd)
# os.system(cmd)

# Create a reduced set of streamlines for easy visualization
reduced_track_out = os.path.join(dest_dwi_dir, 'reduced-tracks_%s_%s_t1.tck' % (subject, session))
cmd = 'tckedit %s -number 200k %s' % (track_out, reduced_track_out)
# print(cmd)
# os.system(cmd)
# View using the mrview command:
# cmd = 'mrview %s -tractography.load %s' % (eddy_mif, reduced_track_out)
# os.system(cmd)
# The color scheme is:
# Red = L-R, Green = P-A, Blue = I-S

# Start work on building the connectome

# Remove over- and under-fitted white matter tracts
out_mu = os.path.join(dest_dwi_dir, 'sift-mu_%s_%s.txt' % (subject, session))
out_coeff = os.path.join(dest_dwi_dir, 'sift-coeffs_%s_%s.txt' % (subject, session))
out_sift = os.path.join(dest_dwi_dir, 'sift-track_%s_%s.txt' % (subject, session))
cmd = 'tcksift2 -act %s -out_mu %s -out_coeffs %s %s %s %s' % (seg_out_coreg, out_mu, out_coeff, track_out, wm_fod, out_sift)
# print(cmd)
# os.system(cmd)

# recon-all has been run on all subjects already, using one of the output files (aparc+aseg.mgz) here
# Converting labels
fs_label = glob.glob('%s/aparc+aseg.mgz' % (dest_anat_dir))[0]
fs_clut = glob.glob('%s/FreeSurferColorLUT.txt' % (dest_dwi_dir))[0]
mrtrix_clut = '/home/sshankar/miniconda3/share/mrtrix3/labelconvert/fs_default.txt'
parcels = os.path.join(dest_dwi_dir, '%s_%s_parcels.mif' % (subject, session))
cmd = 'labelconvert %s %s %s %s' % (fs_label, fs_clut, mrtrix_clut, parcels)
# print(cmd)
# os.system(cmd)

# Coregister the parcellation
coreg_parcel = os.path.join(dest_dwi_dir, 'coreg_%s_%s_parcels.mif' % (subject, session))
cmd = 'mrtransform %s -linear %s -inverse -datatype uint32 %s' % (parcels, transform_mat_mrtrix, coreg_parcel)
# print(cmd)
# os.system(cmd)

# Creating the connectome
coreg_parcel_csv = os.path.join(dest_dwi_dir, 'coreg_%s_%s_parcels.csv' % (subject, session))
inverse_coreg_parcel = os.path.join(dest_dwi_dir, 'inv-coreg_%s_%s_parcels.csv' % (subject, session))
cmd = 'tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in %s %s %s %s -out_assignment %s' % (out_sift, track_out, coreg_parcel, coreg_parcel_csv, inverse_coreg_parcel)
# print(cmd)
# os.system(cmd)

# Plot the connectome
img = np.loadtxt(coreg_parcel_csv, delimiter=',')
plt.figure(figsize=(12, 12))
plt.imshow(img, interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.Reds_r)
plt.colorbar()
plt.show()

# To view endpoints of fibre tracts, use tckmap to create endpoints
tckresample -endpoints reduced-tracks_sub-04_ses-08_t1.tck reduced_endpoints.tck
mrview b0_brain_eddy_dn_sub-04_ses-08_dwi.mif -tractography.load reduced_endpoints.tck -overlay.load gm-fod_sub-04_ses-08_dwi.mif &

# To view streamlines creating a more reduced track file
tckedit tracks_sub-04_ses-08_t1.tck -number 100k reduced-tracks-100k_sub-04_ses-08_t1.tck
