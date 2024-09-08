# Scrip to transform directionless negative pvalues into bidirectional map

from nilearn.image import load_img
import os
import numpy as np
from nilearn.plotting import plot_stat_map,view_img
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiMasker

working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/SecondLevel_permutations/group_01'

img_p = load_img(os.path.join(working_dir,'run_1/righthighlowPmod/righthighlowPmod_logp_max_size.nii.gz'))
img_t = load_img(os.path.join(working_dir,'run_1/righthighlowPmod/righthighlowPmod_t.nii.gz'))

neg_log_pvals = img_p.get_fdata()
t_scores = img_t.get_fdata()

threshold = -np.log10(0.05)

# Create a mask for significant p-values
significant_mask = neg_log_pvals > threshold

# Create a directional p-value map by multiplying significant p-values by the sign of the corresponding t-values
directional_pvals = np.zeros_like(neg_log_pvals)
directional_pvals[significant_mask] = neg_log_pvals[significant_mask] * np.sign(t_scores[significant_mask])

# Save the directional p-value map as a NIfTI image
from nilearn.image import new_img_like
directional_pvals_img = new_img_like(img_p, directional_pvals)

bwr = plt.cm.bwr

# Plot the directional p-value map
plot_stat_map(directional_pvals_img, display_mode='mosaic',title="Directional p-value map", threshold=threshold, cmap=bwr)
plt.show()


inter_view = view_img(directional_pvals_img, threshold=threshold,cmap=bwr)
inter_view.open_in_browser()