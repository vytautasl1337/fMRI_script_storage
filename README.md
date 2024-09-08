Safe storage scripts used for the analysis of fMRI data (first and second level). It is usable, but I will make sure to have better organized repository.

Physio contains scripts to extract physiological information from your BIDS files, assuming there are any.

first_level - first level scripts including regular nilearn commands, as well as more advanced using parametric modulators. 

secondlevel - second level scripts. This is a bit of a mess (and I am out of time to clean it up). Besides regular second level analysis, it involves figure preparation scripts, mixed linear model, ROI, PCA analysis (some might be not finished)

FD_plot - quality control, visualizing framewise displacement in BIDS subjects.

QC - quality control, visualizing first level analysis results, focused on looking for weird crops, or failed coregistration, leading to subject exclusion. 