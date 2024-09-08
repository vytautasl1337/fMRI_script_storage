
#%% Both hands
import numpy as np
import pandas as pd
#from pymer4.models import Lmer # does not work with current R blunder
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
            high_reward_left_file = os.path.join(working_dir, 
                f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-gohighleft_stat-effect_size_statmap.nii.gz')
            high_reward_right_file = os.path.join(working_dir, 
                f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-gohighright_stat-effect_size_statmap.nii.gz')
            low_reward_left_file = os.path.join(working_dir, 
                f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-golowleft_stat-effect_size_statmap.nii.gz')
            low_reward_right_file = os.path.join(working_dir, 
                f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-golowright_stat-effect_size_statmap.nii.gz')
            
            nifti_files.extend([high_reward_left_file, high_reward_right_file, low_reward_left_file, low_reward_right_file])

# Get behavioral data
beh_file = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
ids_to_delete = [1, 8, 36]
beh_ref = beh_ref[~beh_ref['id'].isin(ids_to_delete)]

beh_file.loc[:, 'gender'] = beh_file['gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Create slope table for each condition
behavioral_data = []
for id_, row in beh_ref.iterrows():
    behav_code = row[1]
    sub_data = beh_file[(beh_file['id'] == behav_code) & (beh_file['Correct'] == 1) & (beh_file['GripResponse'] != 0) & (beh_file['Mirror'] == 0)]
    
    for run in range(3):
            
        for rew in range(2):
            
            for hand in range(-1,2,2):
                
                if rew == 0:
                    slope = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['Run'] == run+1) & (sub_data['GripResponse'] == hand)]['Slope'].mean()
                    rew_condition = 1
                    hand_condition = hand
                elif rew == 1:
                    slope = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['Run'] == run+1) & (sub_data['GripResponse'] == hand)]['Slope'].mean()
                    rew_condition = -1
                    hand_condition = hand

                    
                age = sub_data['age'].mean()
                gender = sub_data['gender'].mean()
                ehi = sub_data['EHI'].mean()
                behavioral_data.append({'SubjectID': id_+1, 
                                        'Run': run+1, 
                                        'Reward': rew_condition, 
                                        'Grip':hand_condition,
                                        'Slope': slope,
                                        'Age': age,
                                        'Gender': gender,
                                        'EHI':ehi})

behavioral_df = pd.DataFrame(behavioral_data)

# Extract ROI values
from nilearn.datasets import fetch_atlas_difumo
import nilearn
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img,index_img,threshold_img,math_img
from nilearn.image import load_img
from nilearn import plotting

atlas = fetch_atlas_difumo(512)
maps = load_img(atlas.maps)
medialPFC = index_img(maps,50)

ref_volume = load_img(nifti_files[0])

resampled_mask = resample_to_img(medialPFC, ref_volume,interpolation='nearest')

thresholded_mask = threshold_img(resampled_mask,threshold=0,two_sided=False)
binary_mask = math_img('img > 0', img=thresholded_mask)


masker = NiftiMasker(mask_img=binary_mask)

roi_means = []

for nifti_file in nifti_files:
    
    img = load_img(nifti_file)
    
    roi_data = masker.fit_transform(img)
    
    roi_mean = np.mean(roi_data)
    
    roi_means.append(roi_mean)

behavioral_df['ROI'] = roi_means

from nilearn.plotting import plot_design_matrix
plot_design_matrix(behavioral_df)
plt.show()

behavioral_df.to_excel("roi_lmm.xlsx", index=False)

