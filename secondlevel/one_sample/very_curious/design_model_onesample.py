import os
import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from nilearn.plotting import plot_design_matrix
from nilearn import plotting

def design(contrast1,contrast,save_folder,exclude_list, 
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
    for b in range(beh_ref.shape[0]):

        behav_code = beh_ref.iloc[b,1]
        
        sub_code = participants_list.iloc[b,0]
        
        #sub_data = beh_mri[(beh_mri.Run == 3) & (beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.GripResponse != 0)]
        sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.GripResponse != 0)]
        
        #if behavioral=='Slope':
        sub_data[behavioral] = np.log(sub_data[behavioral])
            
        if cond == 'highwin>highloss':
            beh1 = sub_data[sub_data['RewardReceived'] == 20][behavioral].mean()
            beh2 = sub_data[sub_data['RewardReceived'] == -5][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'highwin>lowloss':
            beh1 = sub_data[sub_data['RewardReceived'] == 20][behavioral].mean()
            beh2 = sub_data[sub_data['RewardReceived'] == -20][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'lowloss>highwin':
            beh1 = sub_data[sub_data['RewardReceived'] == -20][behavioral].mean()
            beh2 = sub_data[sub_data['RewardReceived'] == 20][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'lowloss>lowwin':
            beh1 = sub_data[sub_data['RewardReceived'] == -20][behavioral].mean()
            beh2 = sub_data[sub_data['RewardReceived'] == 5][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'GoHigh>GoLow':
            beh1 = sub_data[sub_data['RewardPromise'] == 1][behavioral].mean()
            beh2 = sub_data[sub_data['RewardPromise'] == -1][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'Go':
            beh12 = sub_data[behavioral].mean()
        if cond == 'right go high':
            beh12 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
        if cond == 'right go low':
            beh12 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
        if cond == 'left go high':
            beh12 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
        if cond == 'left go low':
            beh12 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
        if cond == 'Right high > right low':
            beh1 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
            beh2 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == 1)][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'Left high > left low':
            beh1 = sub_data[(sub_data['RewardPromise'] == 1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
            beh2 = sub_data[(sub_data['RewardPromise'] == -1) & (sub_data['GripResponse'] == -1)][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'right>left':
            beh1 = sub_data[sub_data['GripResponse'] == 1][behavioral].mean()
            beh2 = sub_data[sub_data['GripResponse'] == -1][behavioral].mean()
            beh12 = beh1-beh2
        if cond == 'right':
            beh12 = sub_data[sub_data['GripResponse'] == 1][behavioral].mean()
        if cond == 'left':
            beh12 = sub_data[sub_data['GripResponse'] == -1][behavioral].mean()

        
        ehi_mean = sub_data['EHI'].mean()
        age_mean = sub_data['age'].mean()
        gender_mean = sub_data['gender'].mean()

        dictio = {'SubNr':sub_code,'Code':behav_code,'BEH':beh12, 'age':age_mean, 'ehi':ehi_mean, 'gender':gender_mean}
        
        data_list.append(dictio)
        
    df_clean = pd.DataFrame(data_list)
    
    # find behavioral NaNs, some participants were lazy to fill those in, updates exclude_list
    missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['BEH'])].unique()
    exclude_list.extend(set(missing_data) - set(exclude_list))

    # Now clean matrix
    beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
    beh=beh.iloc[:,2:]
    beh = beh.reset_index(drop=True)
    
    # beh_sorted = beh.sort_values(by='BEH')
    # plt.bar(range(len(beh_sorted)), beh_sorted['BEH'])
    # plt.show()
    
    #Ortho
    # if contrast[0] in [1, -1]:
    #     beh_z = beh-beh.mean()
    # else:
    #     # for column in beh.columns[1:]:
    #     #     beh[column] = beh[column] - beh[column].mean()
    #     beh_z = beh
    beh_z = beh-beh.mean()
    print(beh_z.to_string())
    

    second_level_input = []
    for sub in range(len(participants_list)):
        subject_id = participants_list.participant_id[sub]
        if subject_id not in exclude_list:
            stat_map1 = os.path.join(working_dir,
                                'derivatives/task_output/FirstLevel/FirstLevel_curious_2/{}/first_level_maps/{}_contrast-{}_stat-effect_statmap.nii.gz'.format(subject_id,subject_id,contrast1))
            # stat_map2 = os.path.join(working_dir,
            #                     'derivatives/task_output/FirstLevel/{}/first_level_maps/{}_contrast-{}_stat-z_statmap.nii.gz'.format(subject_id,subject_id,contrast2))
            
            second_level_input.append(stat_map1)
            #second_level_input2.append(stat_map2)

    
    n_subjects = len(second_level_input)
    
    # beh_z_beh1 = np.concatenate([beh_z['BEH1'], beh_z['BEH2']])
    # beh_z_age  = np.concatenate([beh_z['age'], beh_z['age']])
    # beh_z_ehi = np.concatenate([beh_z['ehi'], beh_z['ehi']])
    # beh_z_gender = np.concatenate([beh_z['gender'], beh_z['gender']])
    
    #print(beh_z_beh1)
    
    beh_1 = np.array(beh_z['BEH']).reshape(-1, 1)
    #beh_2 = np.array(beh_z_beh2).reshape(-1, 1)
    age_ = np.array(beh_z['age']).reshape(-1, 1)
    ehi_ = np.array(beh_z['ehi']).reshape(-1, 1)
    gender_ = np.array(beh_z['gender']).reshape(-1, 1)
    #bis_ = np.array(beh_z['bis']).reshape(-1, 1)
    
    condition_effect1 = np.hstack(([1] * n_subjects))
    #condition_effect2 = np.hstack(([0] * n_subjects, [1] * n_subjects))
    
    #subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
    #subjects = [f"S{i:01d}" for i in range(1, n_subjects + 1)]
    
    
    design_matrix = pd.DataFrame(
    np.hstack((condition_effect1[:, np.newaxis],
               beh_1,age_,ehi_,gender_)),
    columns=[contrast1,behavioral,'age','ehi','gender'],
    )
    
    from nilearn import plotting
    dm = plotting.plot_design_matrix(design_matrix)
    #plotting.show()
    
    fig = dm.get_figure()
    fig.savefig(os.path.join(save_folder,'{}_design_matrix.png'.format(label)))
    
    return second_level_input,design_matrix,n_subjects