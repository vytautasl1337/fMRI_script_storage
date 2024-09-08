
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


#contrasts = ['gohighleftPmod','gohighrightPmod','golowleftPmod','golowrightPmod']
contrast = 'nogo'


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


second_level_input = []

for sub in range(len(participants_list)):
    subject_id = participants_list.participant_id[sub]
    if subject_id not in exclude_list:
        stat_map1 = os.path.join(working_dir,
        'derivatives/task_output/FirstLevel_nopmods/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrast))
        second_level_input.append(stat_map1)
        

        

beh_mri = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202406131123/behavioral_group_output_withreject.xlsx')
beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else 0)

data_list=[]

for b in range(beh_ref.shape[0]):

    behav_code = beh_ref.iloc[b,1]
    
    sub_code = participants_list.iloc[b,0]
    

    sub_data = beh_mri[(beh_mri.id == behav_code)]

    
    ehi_mean = sub_data['EHI'].mean()
    age_mean = sub_data['age'].mean()
    gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,
            'Code':behav_code,
            'ehi':ehi_mean,
            'age':age_mean,
            'sex':gender_mean}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)

missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['age'])].unique()
exclude_list.extend(set(missing_data) - set(exclude_list))
# Now clean matrix
beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
beh.reset_index(drop=True, inplace=True)

beh=beh.iloc[:,2:]
beh = beh.reset_index(drop=True)
beh_z = beh-beh.mean()


age_ = np.array(beh_z['age']).reshape(-1, 1)
ehi_ = np.array(beh_z['ehi']).reshape(-1, 1)
sex_ = np.array(beh_z['sex']).reshape(-1, 1)

n_subjects = len(second_level_input)
subject_effect = np.vstack((np.eye(n_subjects)))
subjects = [f"S{i:02d}" for i in range(1, n_subjects + 1)]

intercept = np.hstack(([1] * n_subjects))

design_matrix = pd.DataFrame(
    np.hstack((intercept[:, np.newaxis],
               ehi_,age_,sex_,subject_effect)),
    
    columns=['intercept',
             'ehi','age','gender']+subjects,
)

# design_matrix = pd.DataFrame(
#     intercept[:, np.newaxis],
    
#     columns=['intercept'],
# )


from nilearn.plotting import plot_design_matrix
plot_design_matrix(design_matrix=design_matrix)
plt.show()

from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import resample_to_img
mask_ = fetch_icbm152_2009()

mask = resample_to_img(mask_['mask'],second_level_input[0],interpolation='nearest')

second_level_model = SecondLevelModel(mask_img=mask)
second_level_model = second_level_model.fit(
    second_level_input,
    design_matrix=design_matrix,
)


#bwr = plt.cm.bwr

#contrast_input = [1,0,0,0]

map_ = second_level_model.compute_contrast('intercept')

from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(
    map_, alpha=0.001, height_control="fpr",
)

plotting.plot_stat_map(
    map_,
    threshold=threshold,
    #cut_coords=(2,-6,62),
    colorbar=True,
    display_mode="ortho",
    draw_cross = False,
    black_bg=False,
    #cmap=bwr
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


















