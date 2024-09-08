import numpy as np
import pandas as pd
from pymer4.models import Lmer
import os
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib

# Load participant file
participant_file = pd.read_csv('/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/participants.tsv', sep='\t')
exclude = ['sub-01', 'sub-08', 'sub-036']
working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/FirstLevel_slope/'

# Get first level nifti data
nifti_files = []
for participant_id in participant_file['participant_id']:
    if participant_id not in exclude:
        for run in range(1, 4):
            high_reward_file = os.path.join(working_dir, f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-gohigh_stat-effect_size_statmap.nii.gz')
            low_reward_file = os.path.join(working_dir, f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-golow_stat-effect_size_statmap.nii.gz')
            nifti_files.extend([high_reward_file, low_reward_file])

# Get behavioral data
beh_file = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
ids_to_delete = [1, 8, 36]
beh_ref = beh_ref[~beh_ref['id'].isin(ids_to_delete)]

# Create slope table for each condition
behavioral_data = []
for id_, row in beh_ref.iterrows():
    behav_code = row[1]
    sub_data = beh_file[(beh_file['id'] == behav_code) & (beh_file['Correct'] == 1) & (beh_file['GripResponse'] != 0) & (beh_file['Mirror'] == 0)]
    
    for run in range(3):
        for rew in range(2):
            if rew == 0:
                slope = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['Run'] == run+1)]['Slope'].mean()
                condition = 1
            elif rew == 1:
                slope = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['Run'] == run+1)]['Slope'].mean()
                condition = -1

            behavioral_data.append({'Participant ID': id_+1, 'Run Number': run+1, 'Reward': condition, 'Averaged Slope': slope})

behavioral_df = pd.DataFrame(behavioral_data)

# Load fMRI maps
import nilearn
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.image import load_img
from nilearn import plotting

mask = datasets.load_mni152_gm_mask()
refimg = load_img(nifti_files[0])
resampled_mask = resample_to_img(mask,refimg,interpolation='nearest')
masker = NiftiMasker(mask_img=resampled_mask, standardize=True)
fmri_data = masker.fit_transform(nifti_files)
fmri_data = fmri_data.astype(np.float16)

fmri_df = pd.DataFrame(fmri_data, columns=[f"Voxel_{i}" for i in range(fmri_data.shape[1])])

merged_df = pd.concat([behavioral_df, fmri_df], axis=1)

formula = "Averaged_Slope ~ Run + Reward * " + " * ".join([f"Voxel_{i}" for i in range(190468)]) + " + (1 | Participant_ID)"

model = Lmer(formula, data=merged_df)
results = model.fit()

print(results.summary())