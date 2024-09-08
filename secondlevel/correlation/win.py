#######################################################################################################
#IMPORT LIBRARIES
import pandas as pd
import os,ast,glob,shutil
import numpy as np
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from nilearn.plotting import plot_design_matrix
from nilearn import plotting

working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'
    
participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)

#################################################################################
signif = .001
correction = 'fpr'
run = 'run_1'
exclude_list = ['sub-01','sub-08','sub-036']
contrast1='highwin'


images_dir = os.path.join(working_dir,'derivatives/task_output/FirstLevel_0509/{}/ses-PRISMA',
                          run,'first_level_maps')

# Substract image maps
maps=[]
for i in participants_list.participant_id:
    if i not in exclude_list:
        stat_map = os.path.join(images_dir.format(i),
                                '{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(i,contrast1))
        maps.append(stat_map)
  

n_subjects = len(maps)

# Get additional behavior data
beh_dir = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/'
beh_ref_dir = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/'
beh_results = pd.read_excel(os.path.join(beh_dir,'behavioral_group_output_withreject.xlsx'))
beh_ref = pd.read_excel(os.path.join(beh_ref_dir,'participant_list.xlsx'))

beh_mri = beh_results[beh_results.group == 1] #MRI data
beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else 0)

data_list=[]
for b in range(beh_ref.shape[0]):

    behav_code = beh_ref.iloc[b,1]
    
    sub_code = participants_list.iloc[b,0]
    
    sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&
                       (beh_mri.Mirror != 1) & (beh_mri.GripResponse != 0)]

    beh12 = sub_data[sub_data['RewardPromise'] == 1]['Slope'].mean()
    #beh2 = sub_data[sub_data['RewardPromise'] == -1]['Slope'].mean()
    #beh1 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['RewardReceived'].isin([5,20]))]['Slope'].mean()
    #beh2 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['RewardReceived'].isin([-5,-20]))]['Slope'].mean()
    #beh12 = beh1-beh2
    #beh12 = sub_data[sub_data['RewardReceived'].isin([20,5])]['Slope'].mean()
    #beh2 = sub_data[sub_data['RewardReceived'].isin([-20,-5])]['Slope'].mean()
    #beh12=beh1-beh2
    #beh12 = sub_data['Slope'].mean()
    
    ehi_mean = sub_data['EHI'].mean()
    mdi_mean = sub_data['MDI'].mean()
    bisbas_bas_mean = sub_data['BISBAS_bas'].mean()
    age_mean = sub_data['age'].mean()
    gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,
              'Code':behav_code,
              #'Slope_high':beh1,
              #'Slope_low':beh2,
              'Slope':beh12,
              'age':age_mean,
              'mdi':mdi_mean,
              'bisbas_bas':bisbas_bas_mean,
              'gender':gender_mean}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)

# find behavioral NaNs, some participants were lazy to fill those in, updates exclude_list
missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['age'])].unique()
exclude_list.extend(set(missing_data) - set(exclude_list))
# Now clean matrix
beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
beh=beh.iloc[:,2:]
beh = beh.reset_index(drop=True)
beh_z = beh-beh.mean()

condition_effect = np.hstack(([1] * n_subjects))
beh_ = np.array(beh_z['Slope']).reshape(-1, 1)
beh1_ = np.array(beh_z['Slope_high']).reshape(-1, 1)
beh2_ = np.array(beh_z['Slope_low']).reshape(-1, 1)
age_ = np.array(beh_z['age']).reshape(-1, 1)
#ehi_ = np.array(beh_z['ehi']).reshape(-1, 1)
gender_ = np.array(beh_z['gender']).reshape(-1, 1)
mdi_ = np.array(beh_z['mdi']).reshape(-1, 1)
#shaps_ = np.array(beh_z['shaps']).reshape(-1, 1)
bisbas_bas_ = np.array(beh_z['bisbas_bas']).reshape(-1, 1)
#bis11_ = np.array(beh_z['bis11']).reshape(-1, 1)

design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis],
               beh_,
               #age_, 
               #gender_,
               mdi_,
               bisbas_bas_)),
    columns=['Maps',
             'Slope',
             #'Age',
             #'Gender',
             'MDI',
             'BISBAS_bas'],
)

# design_matrix = pd.DataFrame(
#     np.hstack((condition_effect[:, np.newaxis],
#                #beh_,
#                mdi_)),
#     columns=['Maps',
#              #'Slope',
#              'MDI'],
# )

# plot_design_matrix(design_matrix)
# plt.show()
# plt.close()

from nilearn import datasets
from nilearn.image import load_img
bg = datasets.load_mni152_template(resolution=1)
#mask = load_img('/mnt/projects/PBCTI/ds-MRI-study/templates/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
mask = datasets.load_mni152_gm_mask()

second_level_model_ = SecondLevelModel(mask_img=mask).fit(
    maps, design_matrix=design_matrix
)

stat_map = second_level_model_.compute_contrast(
    'Slope', output_type="z_score"
)


from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(stat_map, 
                                   alpha=signif, 
                                   height_control=correction)
print('Threshold is ', threshold)

plotting.plot_stat_map(
    stat_map,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg = False,
)
plt.show()

inter_view = plotting.view_img(stat_map, threshold=threshold, bg_img = bg)
inter_view.open_in_browser()



















