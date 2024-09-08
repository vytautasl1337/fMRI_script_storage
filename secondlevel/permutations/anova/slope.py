
#######################################################################################################
#IMPORT LIBRARIES
from tkinter import *
import pandas as pd
import os,ast
import numpy as np
from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from nilearn.plotting import plot_design_matrix
from nilearn import plotting

working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'
#################################################################################
session='ses-PRISMA'

run='run_123'


contrasts = ['gohighleftPmod','gohighrightPmod','golowleftPmod','golowrightPmod']
#contrasts = ['gohighright','gohighleft','golowright','golowleft']


# exclude participants
exclude_list = ['sub-08','sub-036']
#+right handers
#exclude_list = ['sub-01','sub-08','sub-036','sub-012','sub-015','sub-025','sub-030']
#####################################################################################################


participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)
# from nilearn import datasets
# bg = datasets.load_mni152_template(resolution=1)
# mask = datasets.load_mni152_gm_mask()

#from nilearn.glm.second_level import non_parametric_inference
# Create design and model for each contrast:


second_level_input1,second_level_input2,second_level_input3,second_level_input4 = [],[],[],[]

for sub in range(len(participants_list)):
    subject_id = participants_list.participant_id[sub]
    if subject_id not in exclude_list:
        stat_map1 = os.path.join(working_dir,
        'derivatives/task_output/FirstLevel_pmod_trueslope/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[0]))
        second_level_input1.append(stat_map1)
        
        stat_map2 = os.path.join(working_dir,
        'derivatives/task_output/FirstLevel_pmod_trueslope/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[1]))
        second_level_input2.append(stat_map2)
        
        stat_map3 = os.path.join(working_dir,
        'derivatives/task_output/FirstLevel_pmod_trueslope/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[2]))
        second_level_input3.append(stat_map3)
        
        stat_map4 = os.path.join(working_dir,
        'derivatives/task_output/FirstLevel_pmod_trueslope/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[3]))
        second_level_input4.append(stat_map4)
        
second_level_input = second_level_input1+second_level_input2+second_level_input3+second_level_input4

beh_mri = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202406131123/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else -1)

data_list=[]
for i in range(4):
    for b in range(beh_ref.shape[0]):

        behav_code = beh_ref.iloc[b,1]
        
        sub_code = participants_list.iloc[b,0]
        
        if i == 0:
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&(beh_mri.RewardPromise==1)&(beh_mri.GripResponse==1)&
                            (beh_mri.Mirror != 1)]
        elif i==1:
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&(beh_mri.RewardPromise==1)&(beh_mri.GripResponse==-1)&
                            (beh_mri.Mirror != 1)]
        elif i==2:
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&(beh_mri.RewardPromise==-1)&(beh_mri.GripResponse==1)&
                            (beh_mri.Mirror != 1)]
        elif i==3:
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&(beh_mri.RewardPromise==-1)&(beh_mri.GripResponse==-1)&
                            (beh_mri.Mirror != 1)]

        slope = sub_data['Slope'].mean()
        rt = sub_data['RT_init'].mean()
        maxgrip = sub_data['MaxGrip'].mean()
        #griplength = sub_data['GripLength'].mean()
        BISBAS_fun = sub_data['BISBAS_fun'].mean()
        BISBAS_rew = sub_data['BISBAS_rew'].mean()
        
        ehi_mean = sub_data['EHI'].mean()
        bisbas_bas_mean = sub_data['BISBAS_bas'].mean()
        bisbas_bis_mean = sub_data['BISBAS_bis'].mean()
        age_mean = sub_data['age'].mean()
        gender_mean = sub_data['gender'].mean()

        dictio = {'SubNr':sub_code,
                'Code':behav_code,
                'Slope':slope,
                'RT':rt,
                'MaxGrip':maxgrip,
                #'GripLength':griplength,
                'BISBAS_fun':BISBAS_fun,
                'BISBAS_rew':BISBAS_rew,
                'ehi':ehi_mean,
                'age':age_mean,
                'sex':gender_mean,
                'bisbas_bas':bisbas_bas_mean,
                'bisbas_bis':bisbas_bis_mean}
            
        data_list.append(dictio)

df_clean = pd.DataFrame(data_list)

missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['age'])].unique()
exclude_list.extend(set(missing_data) - set(exclude_list))
# Now clean matrix
beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
beh.reset_index(drop=True, inplace=True)

beh=beh.iloc[:,2:]
beh = beh.reset_index(drop=True)
beh_z = beh#beh-beh.mean()

slope_ = np.array(beh_z['Slope']).reshape(-1, 1)
RT_ = np.array(beh_z['RT']).reshape(-1, 1)
MaxGrip_ = np.array(beh_z['MaxGrip']).reshape(-1, 1)
#GripLength_ = np.array(beh_z['GripLength']).reshape(-1, 1)
BISBAS_fun_ = np.array(beh_z['BISBAS_fun']).reshape(-1,1)
BISBAS_rew_ = np.array(beh_z['BISBAS_rew']).reshape(-1,1)
rt_ = np.array(beh_z['RT']).reshape(-1, 1)
age_ = np.array(beh_z['age']).reshape(-1, 1)
ehi_ = np.array(beh_z['ehi']).reshape(-1, 1)
sex_ = np.array(beh_z['sex']).reshape(-1, 1)
bisbas_bas_ = np.array(beh_z['bisbas_bas']).reshape(-1, 1)
bisbas_bis_ = np.array(beh_z['bisbas_bis']).reshape(-1, 1)



n_subjects = len(second_level_input1+second_level_input2)
condition_subjects = len(second_level_input1)

high_reward_effect = np.hstack(([1] * n_subjects, [0] * n_subjects))
low_reward_effect = np.hstack(([0] * n_subjects, [1] * n_subjects))

right_hand_effect = np.hstack(([1] * condition_subjects, [0] * condition_subjects, [1] * condition_subjects, [0] * condition_subjects))
left_hand_effect = np.hstack(([0] * condition_subjects, [1] * condition_subjects, [0] * condition_subjects, [1] * condition_subjects))

#interaction = np.hstack(([-1] * condition_subjects, [1] * condition_subjects, [1] * condition_subjects, [-1] * condition_subjects))
intercept = np.hstack(([1] * n_subjects, [1] * n_subjects))

# subject effetcs
subject_effect_ = np.vstack((np.eye(condition_subjects)))
subject_effect = np.vstack([subject_effect_ for _ in range(4)])
subjects = [f"S{i:02d}" for i in range(1, condition_subjects + 1)]


design_matrix = pd.DataFrame(
    np.hstack((high_reward_effect[:, np.newaxis], 
               low_reward_effect[:, np.newaxis],
               right_hand_effect[:, np.newaxis],
               left_hand_effect[:, np.newaxis],
               intercept[:, np.newaxis],#slope_,#RT_,MaxGrip_,
               ehi_,age_,sex_, subject_effect)),
    
    columns=["main effect of high reward",
             "main effect of low reward",
             "main effect of right hand",
             "main effect of left hand",
             'intercept',#'slope',#'RT','MaxGrip',
             'ehi','age','gender']+subjects,
)


from nilearn.plotting import plot_design_matrix
plot_design_matrix(design_matrix=design_matrix)
plt.show()

from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
mask_ = load_mni152_brain_mask()

mask = resample_to_img(mask_,second_level_input1[0],interpolation='nearest')

second_level_model = SecondLevelModel(mask_img=mask)
second_level_model = second_level_model.fit(
    second_level_input,
    design_matrix=design_matrix,
)


bwr = plt.cm.bwr

contrast = [1,-1,0,1,0,0,0,0]+[0]*42

# conjuction
map_ = second_level_model.compute_contrast(contrast, second_level_stat_type='t')

from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    map_, alpha=0.001, height_control="fpr",
)

plotting.plot_stat_map(
    map_,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg=False,
    cmap=bwr
)
plt.show()

inter_view = plotting.view_img(map_, threshold=threshold)
inter_view.open_in_browser()















hand_map = second_level_model.compute_contrast(f_contrast_matrix[1].tolist(), second_level_stat_type='t')

from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    hand_map, alpha=0.001, height_control="fpr",
)

plotting.plot_stat_map(
    hand_map,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg=False,
    cmap=bwr
)
plt.show()



reward_map = second_level_model.compute_contrast(f_contrast_matrix[0].tolist(), second_level_stat_type='t')

from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    reward_map, alpha=0.001, height_control="fpr",
)

plotting.plot_stat_map(
    reward_map,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg=False,
    cmap=bwr
)
plt.show()




interaction_map = second_level_model.compute_contrast(f_contrast_matrix[2].tolist(), second_level_stat_type='t')

from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    interaction_map, alpha=0.1, height_control="fpr",
)

plotting.plot_stat_map(
    interaction_map,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg=False,
    cmap=bwr
)
plt.show()


















