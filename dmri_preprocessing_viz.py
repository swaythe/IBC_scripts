import os
import numpy as np
from dipy.segment.clustering import QuickBundles
from mayavi import mlab
from dipy.viz import colormap

# clustering of fibers into bundles and visualization thereof
derivatives_dir = '/home/sshankar/diffusion/derivatives'
subject = 'sub-04'
session = 'ses-08'
dest_dir = os.path.join(derivatives_dir, subject, session)
dest_dwi_dir = os.path.join(dest_dir, 'dwi')

streamlines_file = os.path.join(dest_dwi_dir, 'streamlines.npz')
streamlines = np.load(streamlines_file)['arr_0']

qb = QuickBundles(threshold=10.)
clusters = qb.cluster(streamlines)
centroids = clusters.centroids
np.savez(os.path.join(dwi_dir, 'centroids.npz'), centroids)

# colors = colormap.line_colors(clusters).astype(float)
#
# mlab.figure(bgcolor=(0., 0., 0.))
#
# for streamline, color in zip(clusters, colors):
#     mlab.plot3d(streamline.T[0], streamline.T[1], streamline.T[2],
#                 line_width=1., tube_radius=.5, color=tuple(color))

# figname = streamlines_file[:-3] + 'png'
# mlab.savefig(figname)
# print(figname)
# mlab.close()
