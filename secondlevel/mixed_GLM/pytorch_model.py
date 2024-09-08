import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from nilearn import plotting
import os
import nibabel as nib
from nilearn import datasets
import matplotlib
import matplotlib.pyplot as plt

# Load participant and behavioral data
participant_file = pd.read_csv('/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/participants.tsv', sep='\t')
exclude = ['sub-01', 'sub-08', 'sub-036']
working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/FirstLevel_slope/'

beh_file = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
ids_to_delete = [1, 8, 36]
beh_ref = beh_ref[~beh_ref['id'].isin(ids_to_delete)]

# get first level nifti data
nifti_files = []
for participant_id in participant_file.participant_id:
    if participant_id not in exclude:
        for run in range(1, 2):
            high_reward_file = os.path.join(working_dir,f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-gohigh_stat-effect_size_statmap.nii.gz')
            low_reward_file = os.path.join(working_dir,f'{participant_id}/ses-PRISMA/run_{run}/first_level_maps/{participant_id}_contrast-golow_stat-effect_size_statmap.nii.gz')
            nifti_files.extend([high_reward_file, low_reward_file])
            
behavioral_data=[]
for id_,b in enumerate(range(beh_ref.shape[0])):

    behav_code = beh_ref.iloc[b,1]
        
    sub_data = beh_file[(beh_file.id == behav_code) & (beh_file.Correct == 1) & (beh_file.GripResponse != 0) &
                        (beh_file.Mirror == 0)]
    
    for run in range(1):
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

# Load fMRI data and resample mask
mask = datasets.load_mni152_gm_mask()
refimg = nib.load(nifti_files[0])
resampled_mask = resample_to_img(mask, refimg, interpolation='nearest')

# Load and preprocess fMRI data
masker = NiftiMasker(mask_img=resampled_mask, standardize=True)
fmri_data = masker.fit_transform(nifti_files)
design_matrix = np.column_stack((behavioral_df['Participant ID'], 
                                 behavioral_df['Run Number'],
                                 behavioral_df['Condition'],
                                 fmri_data))
averaged_slope_array = behavioral_df['Averaged Slope'].values
subjects = behavioral_df['Participant ID'].values

# Define PyTorch model
class SlopeBOLDModel(nn.Module):
    def __init__(self):
        super(SlopeBOLDModel, self).__init__()
        self.fc = nn.Linear(fmri_data.shape[1], 1) 
        
    def forward(self, x):
        return self.fc(x)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(fmri_data)
y_train = torch.FloatTensor(averaged_slope_array).view(-1, 1)

# Initialize model, loss function, and optimizer
model = SlopeBOLDModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Extract coefficients from the model
coefficients = model.fc.weight.detach().numpy().flatten()

# Reshape coefficients to match original brain space
coefficients_3d = coefficients.reshape(masker.mask_img_.shape)

padded_coefficients = np.zeros(masker.mask_img_.shape)
padded_coefficients_1d = padded_coefficients[masker.mask_img_.get_fdata() != 0]

# Threshold coefficients
threshold = 0.001
significant_mask = np.abs(padded_coefficients_1d) > threshold

# Plot significant regions

plotting.plot_stat_map(masker.inverse_transform(significant_mask), 
                                 title='Significant Regions based on PyTorch Model',
                                 display_mode='mosaic')
plt.show()

# Save the plot
display.savefig('/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/SecondLevel_MLM/mlm_pytorch.png', dpi=300)
