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


from nilearn import datasets
from nilearn.image import load_img, resample_to_img, threshold_img, math_img
bg = datasets.load_mni152_template(resolution=1)
mask = datasets.load_mni152_gm_mask()


ref_volume = load_img(high_reward_left_file)

resampled_mask = resample_to_img(mask, ref_volume,interpolation='nearest')

thresholded_mask = threshold_img(resampled_mask,threshold=0,two_sided=False)
binary_mask = math_img('img > 0', img=thresholded_mask)

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=binary_mask)
bold_data_list = [masker.fit_transform(nib.load(f)) for f in nifti_files]
bold_data = np.vstack(bold_data_list)
# bold_data_list = []
# for stat_map in nifti_files:
#     img = nib.load(stat_map)
#     bold_img = masker.fit_transform(img)
#     bold_data = bold_img.get_fdata()
#     flat_data = bold_data.flatten()
#     bold_data_list.append(flat_data)

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

#behavioral_df['BOLD'] = bold_data_list
num_voxels = bold_data.shape[1]
results = []
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

for voxel in range(num_voxels):
    voxel_data = bold_data[:, voxel]
    voxel_df = behavioral_df.copy()
    voxel_df['BOLD'] = voxel_data
    
    # Fit mixed effects model
    model = mixedlm("BOLD ~ Slope * Reward + Run", voxel_df, groups=voxel_df["SubjectID"])
    result = model.fit()
    
    # Store the result
    results.append({
        'voxel': voxel,
        'params': result.params,
        'tvalues': result.tvalues,
        'pvalues': result.pvalues
    })
    
# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Reshape results to nifti image format
param_map = np.zeros(binary_mask.shape)
t_map = np.zeros(binary_mask.shape)
p_map = np.ones(binary_mask.shape)

for i, res in results_df.iterrows():
    voxel = res['voxel']
    param_map[np.unravel_index(voxel, binary_mask.shape)] = res['params']['Slope:Reward']
    t_map[np.unravel_index(voxel, binary_mask.shape)] = res['tvalues']['Slope:Reward']
    p_map[np.unravel_index(voxel, binary_mask.shape)] = res['pvalues']['Slope:Reward']

# Create Nifti images from the maps
param_img = nib.Nifti1Image(param_map, affine=binary_mask.affine)
t_img = nib.Nifti1Image(t_map, affine=binary_mask.affine)
p_img = nib.Nifti1Image(p_map, affine=binary_mask.affine)

# Save the images
param_img.to_filename('param_map.nii.gz')
t_img.to_filename('t_map.nii.gz')
p_img.to_filename('p_map.nii.gz')

# Plot the results
from nilearn.plotting import plot_stat_map, view_img

plot_stat_map(param_img, threshold=3.1, display_mode='z', cut_coords=5, title='Slope:Reward Param Map')
view_img(t_img, threshold=3.1, title='Slope:Reward T Map')