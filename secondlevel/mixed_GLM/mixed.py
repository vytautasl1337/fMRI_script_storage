
########################################################################
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nilearn.input_data import NiftiMasker
from nilearn.image import concat_imgs
from nilearn import plotting
import os
import matplotlib
import matplotlib.pyplot as plt

# participant file
participant_file = pd.read_csv('/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/participants.tsv',sep='\t')
exclude = ['sub-01','sub-08','sub-036']
working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/FirstLevel_slope/'

# get first level nifti data
nifti_files = []
for participant_id in participant_file.participant_id:
    if participant_id not in exclude:
        for run in range(1, 4):
            high_reward_file = os.path.join(working_dir,f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-gohigh_stat-effect_size_statmap.nii.gz')
            low_reward_file = os.path.join(working_dir,f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-golow_stat-effect_size_statmap.nii.gz')
            nifti_files.extend([high_reward_file, low_reward_file])
            

# get behavioral data
beh_file = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
ids_to_delete = [1, 8, 36]
beh_ref = beh_ref[~beh_ref['id'].isin(ids_to_delete)]

# create slope table for each condition
behavioral_data=[]
for id_,b in enumerate(range(beh_ref.shape[0])):

    behav_code = beh_ref.iloc[b,1]
        
    sub_data = beh_file[(beh_file.id == behav_code) & (beh_file.Correct == 1) & (beh_file.GripResponse != 0) &
                        (beh_file.Mirror == 0)]
    
    for run in range(3):
        for rew in range(2):
            if rew == 0:
                slope = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['Run'] == run+1)]['Slope'].mean()
                condition = 1
            elif rew == 1:
                slope = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['Run'] == run+1)]['Slope'].mean()
                condition = -1

            dictio = {'Participant ID':id_+1,
                      'Run Number':run+1,
                      'Condition':condition,
                      'Averaged Slope': slope}
            behavioral_data.append(dictio)

    
behavioral_df = pd.DataFrame(behavioral_data)
    
# Load fMRI maps
from nilearn import datasets
mask = datasets.load_mni152_gm_mask()
from nilearn.image import resample_to_img
import nibabel as nib
refimg = nib.load(nifti_files[0])

resampled_mask = resample_to_img(mask, refimg, interpolation='nearest')


masker = NiftiMasker(mask_img=resampled_mask, standardize=True, smoothing_fwhm=8)

fmri_data = masker.fit_transform(nifti_files)
            
print(type(fmri_data))
print(fmri_data_reduced.shape)

fmri_data_reduced = fmri_data.astype(np.float16)

# Create design matrix
design_matrix = np.column_stack((behavioral_df['Run Number'],
                                 behavioral_df['Condition'],
                                 fmri_data_reduced))
design_matrix = sm.add_constant(design_matrix, prepend=False)


averaged_slope_array = behavioral_df['Averaged Slope'].values
subjects = behavioral_df['Participant ID'].values

# Fit mixed linear model
mixed_lm = sm.MixedLM(endog=averaged_slope_array, exog=design_matrix,
                       groups=subjects)

results = mixed_lm.fit()

# Extract coefficients from results
coefficients = results.params[1:]
# Reshape coefficients to match original brain space
coefficients_3d = coefficients.reshape(masker.mask_img_.shape)
# Threshold coefficients
threshold = 0.05
significant_mask = np.abs(coefficients_3d) > threshold
display = plotting.plot_stat_map(masker.inverse_transform(significant_mask), 
                                 title='Significant MLM Regions - GoHigh>GoLow',
                                 display_mode='mosaic')

display.savefig('/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/SecondLevel_MLM/mlm.png',dpi=300)




