import pandas as pd
from scipy.io import loadmat
import os,numpy,re
from nilearn.glm.first_level import make_first_level_design_matrix

import matplotlib
import matplotlib.pyplot as plt

def design_mat(fmri_task,beh_table,events_dir,physio_dir,comb,subject_id,task_label,t_r,
               combination_output,space_label,phys_exclude,confound_dir,working_dir,session):
    
    if not os.path.isdir(os.path.join(combination_output,'designs')):
        savepath = os.makedirs(os.path.join(combination_output,'designs')) 
    
    
    # Construct design matrix for this combination
    design_matrices = []
    for id_, img in enumerate(fmri_task):
        
        # Number of volumes
        n_scans = img.shape[-1]
        # Number of the run
        idx = comb[id_]
        
        events_table = pd.read_table(events_dir.format(subject_id,session,subject_id,session,task_label,idx))
        
        part_file = pd.read_excel('/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx')
    
        digit = int(re.search(r'sub-0(\d+)', subject_id).group(1))
        
        participant = str(part_file.Behav[digit-1])
        subject_beh = beh_table[(beh_table.id==participant) & (beh_table.Run == idx)] 
        
        subject_beh['Correct'] = subject_beh['Correct'].eq(0).astype(int)
        subject_beh.loc[subject_beh['Mirror'] == 1, 'Correct'] = 1
        events_table['correct_trial'] = subject_beh['Correct'].values
        
        ######################################################################################################
        
        ######################
        # REWARD VS PUNISHMENT
        events_table['reward_feedback'].replace({-5: 'Loss', -20: 'Loss', 5: 'Win', 20: 'Win'}, inplace=True)
        
        ############################
        # SEPARATE GO/NOGO RESPONSES WITH REWARD
        go_conditions = []
        for g,row in enumerate(events_table['response_type']):
            if   row == -1:   go_conditions.append('LeftGo'+events_table['trial_type'][g])
            elif row ==  1:   go_conditions.append('RightGo'+events_table['trial_type'][g])
            elif row ==  0:   go_conditions.append('NoGo'+events_table['trial_type'][g])
                
        go_conditions= [item[:-2] for item in go_conditions]
        events_table['go_conditions']=go_conditions
        
        ###############
        # REWARD BLOCKS
        # need to load from matlab files
        reward_data = loadmat(os.path.join(working_dir,'raw/{}/{}/beh/{}_{}_task-{}_run-{}_events.mat'.format(subject_id,session,
                                                                                                            subject_id,session,
                                                                                                            task_label,idx)))
        
        # Add reward block onsets                                                                                                         
        rew_table = pd.DataFrame()
        rew_table['onset']=reward_data['Block_intro_onset'][0]

        # Add reward block type
        reward_conditions=[]
        for g,row in enumerate(events_table['trial_type']):
            if   row[0] ==  'L':   reward_conditions.append(events_table['trial_type'][g][0]+'owReward')
            elif row[0] ==  'H':   reward_conditions.append(events_table['trial_type'][g][0]+'ighReward')
        reward_blocks=[]
        for i in range(0, len(reward_conditions), 4):
            reward_blocks.append(reward_conditions[i]) 
        rew_table['reward_conditions']=reward_blocks
        
        # Add reward block durations
        trial_reward_onset = reward_data['Trial_Reward_Onset'][0]
        last_reward = trial_reward_onset[3::4]
        block_ends = last_reward+1 # add feedback duration - 1s, now this is the real end of the block
        block_durations = block_ends - reward_data['Block_intro_onset'][0]
        
        rew_table['reward_durations'] = block_durations
            
        #################
        # EMOTIONAL FACES
        events_table['face_conditions'] = reward_data['Emotion_of_Distractor'][0]
        events_table['face_conditions'].replace({0: 'Neutral', -1: 'Sad', 1: 'Happy'}, inplace=True)
        
        
        ##########
        ### ERRORS
        errors=[]    
        for g,row in enumerate(events_table['correct_trial']):
            if   row == 0:   errors.append('Correct')
            elif row == 1:   errors.append('Error')
        events_table['correct_trial']=errors
        
        ##########
        ### PMODS of left and right grips (slope by default)
        # slopes=pd.DataFrame()
        # slopes=pd.DataFrame({'slopes': subject_beh['Slope'], 'grips': go_conditions,
        #                      'duration': events_table.duration.tolist(), 'onset':events_table.onset.tolist()})
        
        # left_grip_slopesL = slopes[slopes['grips'] == 'LeftGoL'].copy()
        # left_grip_slopesH = slopes[slopes['grips'] == 'LeftGoH'].copy()
        
        # mean_slopeL = left_grip_slopesL['slopes'].mean(skipna=True)
        # left_grip_slopesL['slopes_demeaned'] = left_grip_slopesL['slopes'] - mean_slopeL
        # mean_slopeH = left_grip_slopesH['slopes'].mean(skipna=True)
        # left_grip_slopesH['slopes_demeaned'] = left_grip_slopesH['slopes'] - mean_slopeH
        
        # right_grip_slopesL = slopes[slopes['grips'] == 'RightGoL'].copy()
        # right_grip_slopesH = slopes[slopes['grips'] == 'RightGoH'].copy()
        
        # mean_slopeL = right_grip_slopesL['slopes'].mean(skipna=True)
        # right_grip_slopesL['slopes_demeaned'] = right_grip_slopesL['slopes'] - mean_slopeL
        # mean_slopeH = right_grip_slopesH['slopes'].mean(skipna=True)
        # right_grip_slopesH['slopes_demeaned'] = right_grip_slopesH['slopes'] - mean_slopeH
        # ###########################################
        # events_left_pmodL = pd.DataFrame({'trial_type': left_grip_slopesL['grips'], 
        #                                  'onset': left_grip_slopesL['onset'],
        #                                 'duration': left_grip_slopesL['duration'],
        #                                 'modulation':left_grip_slopesL['slopes_demeaned']})
        
        # frame_times = numpy.arange(n_scans) * t_r
        # left_go_pmodsL = make_first_level_design_matrix(
        #         frame_times,
        #         events_left_pmodL,
        #         hrf_model='spm',
        #         drift_model=None) 

        # events_right_pmodL = pd.DataFrame({'trial_type': right_grip_slopesL['grips'], 
        #                         'onset': right_grip_slopesL['onset'],
        #                         'duration': right_grip_slopesL['duration'],
        #                         'modulation':right_grip_slopesL['slopes_demeaned']})
        # right_go_pmodsL = make_first_level_design_matrix(
        #         frame_times,
        #         events_right_pmodL,
        #         hrf_model='spm',
        #         drift_model=None) 
        
        # events_left_pmodH = pd.DataFrame({'trial_type': left_grip_slopesH['grips'], 
        #                             'onset': left_grip_slopesH['onset'],
        #                         'duration': left_grip_slopesH['duration'],
        #                         'modulation':left_grip_slopesH['slopes_demeaned']})
        
        # frame_times = numpy.arange(n_scans) * t_r
        # left_go_pmodsH = make_first_level_design_matrix(
        #         frame_times,
        #         events_left_pmodH,
        #         hrf_model='spm',
        #         drift_model=None) 

        # events_right_pmodH = pd.DataFrame({'trial_type': right_grip_slopesH['grips'], 
        #                         'onset': right_grip_slopesH['onset'],
        #                         'duration': right_grip_slopesH['duration'],
        #                         'modulation':right_grip_slopesH['slopes_demeaned']})
        # right_go_pmodsH = make_first_level_design_matrix(
        #         frame_times,
        #         events_right_pmodH,
        #         hrf_model='spm',
        #         drift_model=None) 

        #######################################
        # Prepare conditions, durations, onsets  
        conditions_ = pd.concat([events_table.reward_feedback,
                                 events_table.go_conditions,
                                 events_table.face_conditions,
                                 ],axis=0)
        
        duration_ = pd.concat([events_table.reward_duration,
                               events_table.duration,
                               events_table.duration,
                               ],axis=0)
        
        onsets_ = pd.concat([events_table.reward_onset,
                             events_table.onset,
                             events_table.onset,
                             ],axis=0)
        


        events = pd.DataFrame({'trial_type': conditions_, 'onset': onsets_,
                        'duration': duration_})
        
        
        # Get confounds
        confound_table = pd.read_csv(confound_dir.format(subject_id,session,subject_id,session,task_label,idx), sep='\t')
        
        confounds = confound_table[['framewise_displacement',
                                    'trans_x','trans_y','trans_z',
                                    'rot_x','rot_y','rot_z']].copy()
        
        # demean?
        confounds = confounds - confounds.mean()
        # Add respiratory/pulse data and merge all confounds and errors
        if subject_id not in phys_exclude:
            physio_file = loadmat(physio_dir.format(subject_id,subject_id,session,task_label,idx))
            data_physio = [[row.flat[0] for row in line] for line in physio_file['R']]
            motion_phys = pd.DataFrame(data_physio)
            # demean?
            #motion_phys = motion_phys - motion_phys.mean()
            motion = pd.concat([motion_phys,confounds],axis=1)
        else:
            motion = confounds
        motion.fillna(0,inplace=True)
        #demean?
        #motion = motion - motion.mean()
        
        # Define the sampling times for the design matrix
        frame_times = numpy.arange(n_scans) * t_r
        
        # Add errors
        error_ = pd.DataFrame({'trial_type': events_table.correct_trial, 'onset': events_table.onset,
                        'duration': events_table.duration})
        confound_matrix = make_first_level_design_matrix(
                frame_times,
                error_,
                hrf_model='spm',
                drift_model='cosine',
                add_regs=motion,
                high_pass = 0.0078125) 
        
        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(confound_matrix)
        plt.show()
        confound_matrix = confound_matrix.drop(columns=['constant', 'Correct'])
        # If no errors occured, create empty Error column
        if 'Error' not in confound_matrix.columns:
            confound_matrix['Error'] = [0]*n_scans
            col_index = confound_matrix.columns.get_loc('Error')
            confound_matrix.insert(0, 'Error', confound_matrix.pop('Error'))

        
        # Finally make design matrix
        design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model='spm',
                drift_model=None)        

        design_matrix = pd.concat([design_matrix,confound_matrix],axis=1)
        #design_matrix = design_matrix.drop(columns=['RightGo', 'LeftGo'])
        # design_matrix['RightGoL_pmod'] = float('nan')
        # design_matrix['RightGoL_pmod'] = right_go_pmodsL['RightGoL']
        # design_matrix['LeftGoL_pmod'] = float('nan')
        # design_matrix['LeftGoL_pmod'] = left_go_pmodsL['LeftGoL']
        # design_matrix['RightGoH_pmod'] = float('nan')
        # design_matrix['RightGoH_pmod'] = right_go_pmodsH['RightGoH']
        # design_matrix['LeftGoH_pmod'] = float('nan')
        # design_matrix['LeftGoH_pmod'] = left_go_pmodsH['LeftGoH']

        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(design_matrix)
        plt.show()
        
        order = ['LeftGoL', 'RightGoL',
                 'LeftGoH', 'RightGoH',
                 'NoGoL','NoGoH','Sad','Neutral','Happy','Win','Loss','Error']
        
        for i in range(6):
            if i in design_matrix.columns:
                design_matrix.rename(columns={i: f'Phys_{i}'}, inplace=True)
                order.append(f'Phys_{i}')

        # Move the remaining columns to the end
        for column in design_matrix.columns:
            if column not in order and type(column)==str and column != 'constant':
                order.append(column)
        order.append('constant')
        design_matrix = design_matrix[order]

        # put the design matrix in a list
        design_matrices.append(design_matrix)
        
        
        from nilearn import plotting
        dm = plotting.plot_design_matrix(design_matrix)
        fig = dm.get_figure()
        fig.savefig(os.path.join(combination_output,'designs','run-{}_design_matrix.png'.format(idx)))
        
    return design_matrices,design_matrix