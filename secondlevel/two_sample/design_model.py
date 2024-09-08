import os
import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from nilearn.plotting import plot_design_matrix
from nilearn import plotting

def design(contrast1,contrast2,contrast,save_folder,exclude_list, 
           participants_list,working_dir,label,behavioral,cond):
    
    beh_dir = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/'
    beh_ref_dir = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/'
    
    beh_results = pd.read_excel(os.path.join(beh_dir,'behavioral_group_output_withreject.xlsx'))
    beh_ref = pd.read_excel(os.path.join(beh_ref_dir,'participant_list.xlsx'))

    #################################################################################

    # Organize behavioral data
    beh_mri = beh_results[beh_results.group == 1] #MRI data

    beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else 0)


    data_list=[]
    if contrast[0] != 0:
        for b in range(beh_ref.shape[0]):
            
            behav_code = beh_ref.iloc[b,1]
            sub_code = participants_list.iloc[b,0]
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.GripResponse != 0)]
            
            ehi_mean = sub_data['EHI'].mean()
            age_mean = sub_data['age'].mean()
            gender_mean = sub_data['gender'].mean()

            dictio = {'SubNr':sub_code,'Code':behav_code, 'age':age_mean, 'ehi':ehi_mean, 'gender':gender_mean}
            
            data_list.append(dictio)
            
    elif contrast[0] == 0:
        
        for b in range(beh_ref.shape[0]):

            behav_code = beh_ref.iloc[b,1]
            
            sub_code = participants_list.iloc[b,0]
            
            sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.GripResponse != 0)]
            
            sub_data[behavioral] = np.log(sub_data[behavioral])
                
            if cond == 'win>loss':
                beh1 = sub_data[sub_data['RewardReceived'].isin([20, 5])][behavioral].mean()
                beh2 = sub_data[sub_data['RewardReceived'].isin([-20, -5])][behavioral].mean()
            elif cond == 'GoHigh>GoLow':
                beh1 = sub_data[sub_data['RewardPromise'] == 1][behavioral].mean()
                beh2 = sub_data[sub_data['RewardPromise'] == -1][behavioral].mean()
            elif cond == 'Right high>right low':
                beh1 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
                beh2 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
            elif cond == 'Left high>left low':
                beh1 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
                beh2 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
            elif cond == 'sad>neutral':
                beh1 = sub_data[sub_data['FaceEmotion'] == -1][behavioral].mean()
                beh2 = sub_data[sub_data['FaceEmotion'] == 0][behavioral].mean()
            elif cond == 'sad>happy':
                beh1 = sub_data[sub_data['FaceEmotion'] == -1][behavioral].mean()
                beh2 = sub_data[sub_data['FaceEmotion'] == 1][behavioral].mean()
            elif cond == 'high>low':
                beh1 = sub_data[sub_data['RewardPromise'] == 1][behavioral].mean()
                beh2 = sub_data[sub_data['RewardPromise'] == -1][behavioral].mean()


        
            ehi_mean = sub_data['EHI'].mean()
            age_mean = sub_data['age'].mean()
            gender_mean = sub_data['gender'].mean()

            dictio = {'SubNr':sub_code,'Code':behav_code,'BEH1':beh1,'BEH2':beh2, 'age':age_mean, 'ehi':ehi_mean, 'gender':gender_mean}
        
            data_list.append(dictio)
        
    df_clean = pd.DataFrame(data_list)
    
    # find behavioral NaNs, some participants were lazy to fill those in, updates exclude_list
    missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['age'])].unique()
    exclude_list.extend(set(missing_data) - set(exclude_list))

    # Now clean matrix
    beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
    beh=beh.iloc[:,2:]
    beh = beh.reset_index(drop=True)
    
    # Ortho
    beh_z = beh-beh.mean()
    print(beh_z.to_string())
    

    second_level_input1,second_level_input2 = [],[]
    for sub in range(len(participants_list)):
        subject_id = participants_list.participant_id[sub]
        if subject_id not in exclude_list:
            stat_map1 = os.path.join(working_dir,
                                'derivatives/task_output/FirstLevel/FirstLevel_final/{}/first_level_maps/{}_contrast-{}_stat-effect_statmap.nii.gz'.format(subject_id,subject_id,contrast1))
            stat_map2 = os.path.join(working_dir,
                                'derivatives/task_output/FirstLevel/FirstLevel_final/{}/first_level_maps/{}_contrast-{}_stat-effect_statmap.nii.gz'.format(subject_id,subject_id,contrast2))
            
            second_level_input1.append(stat_map1)
            second_level_input2.append(stat_map2)

    second_level_input = second_level_input1+second_level_input2
    n_subjects = len(second_level_input1)
    
    if contrast[0] == 0:
        beh1_stack = np.hstack((beh_z['BEH1'], [0] * n_subjects)).reshape(-1, 1)
        beh2_stack = np.hstack(([0] * n_subjects, beh_z['BEH2'])).reshape(-1, 1)
        #beh_z_beh1 = np.concatenate([beh_z['BEH1'], beh_z['BEH2']]).reshape(-1, 1)
        
    beh_z_age  = np.concatenate([beh_z['age'], beh_z['age']]).reshape(-1, 1)
    beh_z_ehi = np.concatenate([beh_z['ehi'], beh_z['ehi']]).reshape(-1, 1)
    beh_z_gender = np.concatenate([beh_z['gender'], beh_z['gender']]).reshape(-1, 1)
    
    condition_effect = np.hstack(([1] * n_subjects, [-1] * n_subjects))
    
    subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
    subjects = [f"S{i:01d}" for i in range(1, n_subjects + 1)]
    
    if contrast[0] != 0:
        design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis],
                beh_z_age,beh_z_ehi,beh_z_gender,subject_effect)),
        columns=[cond,'age','ehi','gender']+subjects,
        )
    else:
        design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis],
                beh1_stack,beh2_stack,beh_z_age,beh_z_ehi,beh_z_gender,subject_effect)),
        columns=[cond,behavioral+'_1',behavioral+'_2','age','ehi','gender']+subjects,
        )
    
    from nilearn import plotting
    dm = plotting.plot_design_matrix(design_matrix)
    #plotting.plot_design_matrix(design_matrix)
    #plotting.show()
    
    fig = dm.get_figure()
    fig.savefig(os.path.join(save_folder,'{}_design_matrix.png'.format(label)))
    
    return second_level_input,design_matrix,n_subjects
