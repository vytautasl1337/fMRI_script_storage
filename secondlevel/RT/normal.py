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

def no_pmods(index,session,p_value,run):
    
    working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'

    contrasts = ['go','rightgo','leftgo',
                'leftright','rightleft',
                'gohigh','golow',
                'gohighright','gohighleft',
                'golowright','golowleft',
                'highlow','lowhigh','righthighlow','lefthighlow']
                
    contrast=contrasts[index-1]
    
    for folder in ['FirstLevel_normal']:#,'FirstLevel_pmod_gripdur']:

        if folder == 'FirstLevel_normal':
            output_folder1 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/normal_noslope/{}/{}/{}'.format(p_value,run,contrast))
            output_folder_non1 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/normal_noslope_nonparam/{}/{}/{}'.format(p_value,run,contrast))
        elif folder == 'FirstLevel_gripdur':
            output_folder1 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/grip_duration_noslope/{}/{}/{}'.format(p_value,run,contrast))
            output_folder_non1 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/grip_duration_noslope_nonparam/{}/{}/{}'.format(p_value,run,contrast))

        if not os.path.isdir(output_folder1):
            savepath1 = os.makedirs(output_folder1) 
        if not os.path.isdir(output_folder_non1):
            savepath2 = os.makedirs(output_folder_non1) 
            
            
        # exclude participants
        exclude_list = ['sub-08','sub-036']
        #####################################################################################################


        participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)

        #####
        # without slope as regressor

        second_level_input = []

        for sub in range(len(participants_list)):
            subject_id = participants_list.participant_id[sub]
            if subject_id not in exclude_list:
                stat_map1 = os.path.join(working_dir,
                'derivatives/task_output/FirstLevel/{}/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(folder,subject_id,session,run,subject_id,contrast))
                second_level_input.append(stat_map1)
                

        n_subjects = len(second_level_input)

        condition_effect = np.hstack(([1] * n_subjects))

        design_matrix = pd.DataFrame(
            condition_effect[:, np.newaxis], columns=["intercept"]
        )

        from nilearn import plotting
        dm = plotting.plot_design_matrix(design_matrix)
        fig = dm.get_figure()
        fig.savefig(os.path.join(output_folder1,'{}_design_matrix.png'.format(contrast)))

        from nilearn.glm.second_level import SecondLevelModel
        from nilearn.datasets import load_mni152_brain_mask
        from nilearn.image import resample_to_img
        mask_ = load_mni152_brain_mask()

        mask = resample_to_img(mask_,second_level_input[0],interpolation='nearest')

        second_level_model = SecondLevelModel(mask_img=mask)
        second_level_model = second_level_model.fit(
            second_level_input,
            design_matrix=design_matrix,
        )


        # conjuction
        map_ = second_level_model.compute_contrast('intercept')

        from nilearn.glm import threshold_stats_img
        _, threshold = threshold_stats_img(
            map_, alpha=p_value, height_control="fpr",
        )

        figure_contrast = plotting.plot_stat_map(
            map_,
            threshold=threshold,
            colorbar=True,
            display_mode="ortho",
            draw_cross = False,
            black_bg=False,
        )
        figure_contrast.savefig(os.path.join(output_folder1,'{}.png'.format(contrast)))


        inter_view = plotting.view_img(map_, threshold=threshold)
        inter_view.save_as_html(os.path.join(output_folder1,'{}.html'.format(contrast)))
        
        from nilearn.glm.second_level import non_parametric_inference
        out_dict = non_parametric_inference(
            second_level_input,
            second_level_contrast='intercept',
            design_matrix=design_matrix,
            mask=mask,
            model_intercept=True,
            n_perm=5000,
            two_sided_test=True,
            n_jobs=10,
            threshold=0.01,
            tfce = False,
        )
        
        
        for i,key in enumerate(out_dict.keys()):
            out_dict[key].to_filename(os.path.join(output_folder_non1,'{}_{}.nii.gz'.format(contrast,key)))


        threshold = -np.log10(0.1) # should be 5 % corrected
        bwr = plt.cm.bwr

        plot1 = plotting.plot_stat_map(
            out_dict["logp_max_size"],
            threshold=threshold,
            colorbar=True,
            display_mode="mosaic",
            draw_cross = False,
            black_bg=False,
            cmap=bwr
        )

        plot1.savefig(os.path.join(output_folder_non1,'clustersize.png'),dpi=300)



        #######
        # with slope

        if folder == 'FirstLevel_normal':
            output_folder2 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/normal_withslope/{}/{}/{}'.format(p_value,run,contrast))
            output_folder_non2 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/normal_withslope_nonparam/{}/{}/{}'.format(p_value,run,contrast))
        elif folder == 'FirstLevel_gripdur':
            output_folder2 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/grip_duration_withslope/{}/{}/{}'.format(p_value,run,contrast))
            output_folder_non2 = os.path.join(working_dir,
                'derivatives/task_output/SecondLevel/grip_duration_withslope_nonparam/{}/{}/{}'.format(p_value,run,contrast))
        if not os.path.isdir(output_folder2):
            savepath1 = os.makedirs(output_folder2) 
        if not os.path.isdir(output_folder_non2):
            savepath2 = os.makedirs(output_folder_non2) 
            
        #####################################################################################################

        participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)


        second_level_input = []

        for sub in range(len(participants_list)):
            subject_id = participants_list.participant_id[sub]
            if subject_id not in exclude_list:
                stat_map1 = os.path.join(working_dir,
                'derivatives/task_output/FirstLevel/{}/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(folder,subject_id,session,run,subject_id,contrast))
                second_level_input.append(stat_map1)
                
                

        beh_mri = pd.read_excel('/mnt/projects/PBCTI/combined/Results/Group_output_202406131123/behavioral_group_output_withreject.xlsx')
        beh_ref = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
        beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else -1)

        data_list=[]
        for b in range(beh_ref.shape[0]):

            behav_code = beh_ref.iloc[b,1]
            
            sub_code = participants_list.iloc[b,0]
            
            if run == 'run_1':
                sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) &(beh_mri.GripResponse!=0) & (beh_mri.Run==1)&
                                    (beh_mri.Mirror != 1)]
            elif run == 'run_123':
                sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) &(beh_mri.GripResponse!=0) &
                                    (beh_mri.Mirror != 1)]
            
            if contrast in ['go']:
                slope = sub_data['Slope'].mean()
                
            elif contrast in ['rightgo']:
                slope = sub_data[(sub_data['GripResponse'] == 1)]['Slope'].mean()
                
            elif contrast in ['leftgo']:
                slope = sub_data[(sub_data['GripResponse'] == -1)]['Slope'].mean()
                
            elif contrast in ['leftright']:
                slope_left = sub_data[(sub_data['GripResponse'] == -1)]['Slope'].mean()
                slope_right = sub_data[(sub_data['GripResponse'] == 1)]['Slope'].mean()
                slope = slope_left-slope_right
                
            elif contrast in ['rightleft']:
                slope_left = sub_data[(sub_data['GripResponse'] == -1)]['Slope'].mean()
                slope_right = sub_data[(sub_data['GripResponse'] == 1)]['Slope'].mean()
                slope = slope_right - slope_left
                
            elif contrast in ['gohigh']:
                slope = sub_data[(sub_data['RewardPromise'] == 1)]['Slope'].mean()
                
            elif contrast in ['golow']:
                slope = sub_data[(sub_data['RewardPromise'] == -1)]['Slope'].mean()
                
            elif contrast in ['gohighright']:
                slope = sub_data[(sub_data['RewardPromise'] == 1)&(sub_data['GripResponse']==1)]['Slope'].mean()
                
            elif contrast in ['gohighleft']:
                slope = sub_data[(sub_data['RewardPromise'] == 1)&(sub_data['GripResponse']==-1)]['Slope'].mean()
                
            elif contrast in ['golowright']:
                slope = sub_data[(sub_data['RewardPromise'] == -1)&(sub_data['GripResponse']==1)]['Slope'].mean()
                
            elif contrast in ['golowleft']:
                slope = sub_data[(sub_data['RewardPromise'] == -1)&(sub_data['GripResponse']==-1)]['Slope'].mean()
                
            elif contrast in ['highlow']:
                slope_high = sub_data[(sub_data['RewardPromise'] == 1)]['Slope'].mean()
                slope_low = sub_data[(sub_data['RewardPromise'] == -1)]['Slope'].mean()
                slope = slope_high - slope_low
                
            elif contrast in ['lowhigh']:
                slope_high = sub_data[(sub_data['RewardPromise'] == 1)]['Slope'].mean()
                slope_low = sub_data[(sub_data['RewardPromise'] == -1)]['Slope'].mean()
                slope = slope_low - slope_high
                
            elif contrast in ['righthighlow']:
                slope_high = sub_data[(sub_data['RewardPromise'] == 1)&(sub_data['GripResponse']==1)]['Slope'].mean()
                slope_low = sub_data[(sub_data['RewardPromise'] == -1)&(sub_data['GripResponse']==1)]['Slope'].mean()
                slope = slope_high - slope_low
                
            elif contrast in ['lefthighlow']:
                slope_high = sub_data[(sub_data['RewardPromise'] == 1)&(sub_data['GripResponse']==-1)]['Slope'].mean()
                slope_low = sub_data[(sub_data['RewardPromise'] == -1)&(sub_data['GripResponse']==-1)]['Slope'].mean()
                slope = slope_high - slope_low
                    
            age_mean = sub_data['age'].mean()
            dictio = {'SubNr':sub_code,
                    'Code':behav_code,
                    'Slope':slope,
                    'age':age_mean
                    }
                
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
                

        n_subjects = len(second_level_input)

        condition_effect = np.hstack(([1] * n_subjects))
        slope_ = np.array(beh_z['Slope']).reshape(-1, 1)


        design_matrix = pd.DataFrame(
            np.hstack((condition_effect[:, np.newaxis],slope_)), columns=["intercept","Slope"]
        )
        # from scipy.stats import zscore
        # design_matrix = design_matrix.apply(zscore)
        # design_matrix = design_matrix.fillna(1)

        from nilearn import plotting
        dm = plotting.plot_design_matrix(design_matrix)
        fig = dm.get_figure()
        fig.savefig(os.path.join(output_folder2,'{}_design_matrix.png'.format(contrast)))

        from nilearn.glm.second_level import SecondLevelModel
        from nilearn.datasets import load_mni152_brain_mask
        from nilearn.image import resample_to_img
        mask_ = load_mni152_brain_mask()

        mask = resample_to_img(mask_,second_level_input[0],interpolation='nearest')

        second_level_model = SecondLevelModel(mask_img=mask)
        second_level_model = second_level_model.fit(
            second_level_input,
            design_matrix=design_matrix,
        )


        # conjuction
        map_ = second_level_model.compute_contrast('Slope')

        from nilearn.glm import threshold_stats_img
        _, threshold = threshold_stats_img(
            map_, alpha=p_value, height_control="fpr",
        )

        figure_contrast = plotting.plot_stat_map(
            map_,
            threshold=threshold,
            colorbar=True,
            display_mode="ortho",
            draw_cross = False,
            black_bg=False,
        )
        figure_contrast.savefig(os.path.join(output_folder2,'{}.png'.format(contrast)))
        

        inter_view = plotting.view_img(map_, threshold=threshold)
        inter_view.save_as_html(os.path.join(output_folder2,'{}.html'.format(contrast)))
        
        from nilearn.glm.second_level import non_parametric_inference
        
        out_dict = non_parametric_inference(
            second_level_input,
            second_level_contrast='Slope',
            design_matrix=design_matrix,
            mask=mask,
            model_intercept=True,
            n_perm=5000,
            two_sided_test=True,
            n_jobs=10,
            threshold=0.01,
            tfce = False,
        )
        
        
        for i,key in enumerate(out_dict.keys()):
            out_dict[key].to_filename(os.path.join(output_folder_non2,'{}_{}.nii.gz'.format(contrast,key)))


        threshold = -np.log10(0.1) # should be 5 % corrected
        bwr = plt.cm.bwr

        plot1 = plotting.plot_stat_map(
            out_dict["logp_max_size"],
            threshold=threshold,
            colorbar=True,
            display_mode="mosaic",
            draw_cross = False,
            black_bg=False,
            cmap=bwr
        )

        plot1.savefig(os.path.join(output_folder_non2,'clustersize.png'),dpi=300)