# Script used after second level analysis

# Plots brain slices in relation to defined regions of interest

# Atlas - DiFuMo

import nilearn.datasets
import nibabel as nib
import matplotlib.pyplot as plt
atlas = nilearn.datasets.fetch_atlas_difumo(dimension=1024)

from nilearn import image
from nilearn import plotting
maps = image.load_img(atlas.maps)

volume = image.index_img(maps,797)



plotting.plot_stat_map(volume)
plt.show()
