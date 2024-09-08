
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

run='run_1'


contrasts = ['gohighleftPmod','gohighrightPmod','golowleftPmod','golowrightPmod']

contr_=[[1,0,0,0],[-1,0,0,0],[0,1,0,0],[0,-1,0,0],[0,0,1,0],[0,0,-1,0]]
#contrast = contr_[0]

for i,contrast in enumerate(contr_):
    contrast_str = ', '.join(map(str, contrast))

    output = os.path.join(working_dir,
                        'derivatives/task_output/SecondLevel_permutations/group_anova_parametric/{}/{}'.format(run,contrast_str))

    # exclude participants
    exclude_list = ['sub-08','sub-036']
    #+right handers
    #exclude_list = ['sub-01','sub-08','sub-036','sub-012','sub-015','sub-025','sub-030']
    #####################################################################################################
    if not os.path.isdir(output):
        print('Creating folder for second level task results...')
        savepath = os.makedirs(output) 
        

    participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)
    from nilearn import datasets
    bg = datasets.load_mni152_template(resolution=1)
    mask = datasets.load_mni152_gm_mask()

    from nilearn.glm.second_level import non_parametric_inference
    # Create design and model for each contrast:

    cont_output = os.path.join(output)
    if not os.path.isdir(cont_output):
        savepath = os.makedirs(cont_output) 
        
    second_level_input1,second_level_input2,second_level_input3,second_level_input4 = [],[],[],[]

    for sub in range(len(participants_list)):
        subject_id = participants_list.participant_id[sub]
        if subject_id not in exclude_list:
            stat_map1 = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0605/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[0]))
            second_level_input1.append(stat_map1)
            
            stat_map2 = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0605/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[1]))
            second_level_input2.append(stat_map2)
            
            stat_map3 = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0605/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[2]))
            second_level_input3.append(stat_map3)
            
            stat_map4 = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0605/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrasts[3]))
            second_level_input4.append(stat_map4)
            
    second_level_input = second_level_input1+second_level_input2+second_level_input3+second_level_input4
    n_subjects = len(second_level_input1+second_level_input2)
    condition_subjects = len(second_level_input1)

    reward_effect = np.hstack(([1] * n_subjects, [-1] * n_subjects))
    hand_effect = np.hstack(([-1] * condition_subjects, [1] * condition_subjects, [-1] * condition_subjects, [1] * condition_subjects))
    interaction = np.hstack(([-1] * condition_subjects, [1] * condition_subjects, [1] * condition_subjects, [-1] * condition_subjects))
    intercept = np.hstack(([1] * n_subjects, [1] * n_subjects))

    design_matrix = pd.DataFrame(
        np.vstack((reward_effect, hand_effect, interaction, intercept)).T,
        columns=["main effect of reward", "main effect of hand", "interaction", "intercept"],
    )

    from nilearn.glm.second_level import SecondLevelModel

    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(
        second_level_input,
        design_matrix=design_matrix,
    )

    z_map = second_level_model.compute_contrast(
        second_level_contrast=contrast,
        output_type="z_score",
        second_level_stat_type = 'F'
    )

    threshold=3.1
    bwr = plt.cm.bwr

    plot1 = plotting.plot_stat_map(
        z_map,
        threshold=threshold,
        colorbar=True,
        display_mode="mosaic",
        draw_cross = False,
        black_bg=False,
        cmap=bwr
    )

    plot1.savefig(os.path.join(cont_output,'clustersize.png'),dpi=300)


    matplotlib.pyplot.close()














