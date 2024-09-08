from brainstat.stats.terms import FixedEffect
from brainstat.stats.terms import MixedEffect
import pandas as pd
import os,ast,glob,shutil
import numpy as np
from nilearn import image
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from nilearn import plotting
from nilearn.plotting import plot_design_matrix
import nibabel as nib


working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'
    
participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)

#################################################################################
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

        
bold_data_list = []
for stat_map in nifti_files:
    bold_img = nib.load(stat_map)
    bold_data = bold_img.get_fdata()
    flat_data = bold_data.flatten()
    bold_data_list.append(flat_data)

np_bold = np.array(bold_data_list)

print(np_bold.shape)


# Get behavior data
beh_dir = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/'
beh_ref_dir = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/'
beh_results = pd.read_excel(os.path.join(beh_dir,'behavioral_group_output_withreject.xlsx'))
beh_ref = pd.read_excel(os.path.join(beh_ref_dir,'participant_list.xlsx'))
ids_to_delete = [1, 8, 36]
beh_ref = beh_ref[~beh_ref['id'].isin(ids_to_delete)]

beh_file = beh_results[beh_results.group == 1] #MRI data
beh_file.loc[:, 'gender'] = beh_file['gender'].apply(lambda x: 1 if x == 'Female' else 0)

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

behavioral_df['Run'] = behavioral_df['Run'].astype(str)
behavioral_df['Reward'] = behavioral_df['Reward'].astype(str)
behavioral_df['Grip'] = behavioral_df['Grip'].astype(str)

term_subject = MixedEffect(behavioral_df.SubjectID)
term_run = MixedEffect(behavioral_df.Run)
term_grip = FixedEffect(behavioral_df.Grip)
term_slope = FixedEffect(behavioral_df.Slope)
term_reward = FixedEffect(behavioral_df.Reward)

model = term_grip + term_slope + term_reward  + term_subject
model = term_reward + term_slope
# BrainStat fixed effects of Slope on gohigh>golow
from brainstat.stats.SLM import SLM

from nilearn import datasets
bg = datasets.load_mni152_template(resolution=1)
mask = datasets.load_mni152_gm_mask()

from nilearn.image import resample_to_img

resampled_mask = resample_to_img(mask,bold_img,interpolation='nearest')

np_mask = resampled_mask.get_fdata()

mask_flat = np_mask.flatten().astype(bool)


contrast_Slope = behavioral_df.Slope
slm_Slope = SLM(
    model,
    -contrast_Slope,
    correction = None,
    mask = mask_flat,
    cluster_threshold=0,
    two_tailed=False
)

slm_Slope.fit(np_bold)


cp = [np.copy(slm_Slope.P["pval"]["C"])]

pp = [np.copy(slm_Slope.P["pval"]["P"])]

qp = [np.copy(slm_Slope.Q)]

[np.place(x, np.logical_or(x > 0, ~mask), np.nan) for x in qp]










