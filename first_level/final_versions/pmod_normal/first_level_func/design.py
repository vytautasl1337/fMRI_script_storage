import pandas as pd
from scipy.io import loadmat
import os,numpy,re
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
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
        
        #################
        # EMOTIONAL FACES
        def map_emotion(trial_type):
            second_letter = trial_type[1]
            if second_letter == 'H':
                return 'Happy'
            elif second_letter == 'S':
                return 'Sad'
            elif second_letter == 'N':
                return 'Neutral'
        events_table['emotions'] = events_table['trial_type'].apply(map_emotion)
        
        # gender is ignored but can be included tweaking the function above
        ##########
        ### ERRORS
        errors=[]    
        for g,row in enumerate(events_table['correct_trial']):
            if   row == 0:   errors.append('Correct')
            elif row == 1:   errors.append('Error')
        events_table['correct_trial']=errors
        
        # Add slope values
        events_table['Slope'] = subject_beh['Slope'].values
        
        ##########
        ### PMODS of left and right grips (slope by default)
        # Separate go conditions to avoid potential nan values
        rightgoL = events_table[events_table['go_conditions'] == 'RightGoL'].dropna().copy()
        rightgoH = events_table[events_table['go_conditions'] == 'RightGoH'].dropna().copy()
        leftgoL = events_table[events_table['go_conditions'] == 'LeftGoL'].dropna().copy()
        leftgoH = events_table[events_table['go_conditions'] == 'LeftGoH'].dropna().copy()
        
        nogoL = events_table[events_table['go_conditions'] == 'NoGoL'].copy()
        nogoH = events_table[events_table['go_conditions'] == 'NoGoH'].copy()
        
        # Substract slope mean
        mean_slope = rightgoL['Slope'].mean()
        rightgoL['slopes_demeaned'] = rightgoL['Slope'] - mean_slope
        
        mean_slope = leftgoL['Slope'].mean()
        leftgoL['slopes_demeaned'] = leftgoL['Slope'] - mean_slope

        mean_slope = rightgoH['Slope'].mean()
        rightgoH['slopes_demeaned'] = rightgoH['Slope'] - mean_slope
        
        mean_slope = leftgoH['Slope'].mean()
        leftgoH['slopes_demeaned'] = leftgoH['Slope'] - mean_slope

        ###########################################
        frame_times = numpy.arange(n_scans) * t_r
        
        events_left_pmodL = pd.DataFrame({'trial_type': leftgoL['go_conditions'], 
                                         'onset': leftgoL['onset'],
                                        'duration': leftgoL['duration'],
                                        'modulation':leftgoL['slopes_demeaned']})
        
        left_go_pmodsL = make_first_level_design_matrix(
                frame_times,
                events_left_pmodL,
                hrf_model='spm',
                high_pass=0) 
        

        events_right_pmodL = pd.DataFrame({'trial_type': rightgoL['go_conditions'], 
                                'onset': rightgoL['onset'],
                                'duration': rightgoL['duration'],
                                'modulation':rightgoL['slopes_demeaned']})
        
        right_go_pmodsL = make_first_level_design_matrix(
                frame_times,
                events_right_pmodL,
                hrf_model='spm',
                high_pass=0) 
        
        events_left_pmodH = pd.DataFrame({'trial_type': leftgoH['go_conditions'], 
                                    'onset': leftgoH['onset'],
                                'duration': leftgoH['duration'],
                                'modulation':leftgoH['slopes_demeaned']})
        left_go_pmodsH = make_first_level_design_matrix(
                frame_times,
                events_left_pmodH,
                hrf_model='spm',
                high_pass=0) 

        events_right_pmodH = pd.DataFrame({'trial_type': rightgoH['go_conditions'], 
                                'onset': rightgoH['onset'],
                                'duration': rightgoH['duration'],
                                'modulation':rightgoH['slopes_demeaned']})
        right_go_pmodsH = make_first_level_design_matrix(
                frame_times,
                events_right_pmodH,
                hrf_model='spm',
                high_pass=0) 

        #######################################
        # Prepare conditions, durations, onsets  
        conditions_ = pd.concat([events_table.reward_feedback,
                                 rightgoH.go_conditions,
                                 rightgoL.go_conditions,
                                 leftgoH.go_conditions,
                                 leftgoL.go_conditions,
                                 nogoL.go_conditions,
                                 nogoH.go_conditions,
                                 events_table.emotions,
                                 ],axis=0)
        
        duration_ = pd.concat([events_table.reward_duration,
                               rightgoH.duration,
                               rightgoL.duration,
                               leftgoH.duration,
                               leftgoL.duration,
                               nogoL.duration,
                               nogoH.duration,
                               events_table.duration,
                               ],axis=0)
        
        onsets_ = pd.concat([events_table.reward_onset,
                             rightgoH.onset,
                             rightgoL.onset,
                             leftgoH.onset,
                             leftgoL.onset,
                             nogoL.onset,
                             nogoH.onset,
                             events_table.onset,
                             ],axis=0)
        

        events = pd.DataFrame({'trial_type': conditions_, 'onset': onsets_,
                        'duration': duration_})
        
        
        # Get confounds
        confound_table = pd.read_csv(confound_dir.format(subject_id,session,subject_id,session,task_label,idx), sep='\t')
        
        cosines = confound_table.filter(like='cosine')
       
        fd_mp = confound_table[['framewise_displacement',
                                    'trans_x','trans_y','trans_z',
                                    'rot_x','rot_y','rot_z']].copy()
        
        fd_mp = fd_mp - fd_mp.mean(skipna=True)
        
        confounds = pd.concat([fd_mp,cosines],axis=1)
        confounds.fillna(0,inplace=True)
        
        
        # Add respiratory/pulse data and merge all confounds and errors
        if subject_id not in phys_exclude:
            physio_file = loadmat(physio_dir.format(subject_id,subject_id,session,task_label,idx))
            data_physio = [[row.flat[0] for row in line] for line in physio_file['R']]
            motion_phys = pd.DataFrame(data_physio)
            motion = pd.concat([motion_phys,confounds],axis=1)
        else:
            motion = confounds
        motion.fillna(0,inplace=True)
                
        # Add errors
        error_ = pd.DataFrame({'trial_type': events_table.correct_trial, 'onset': events_table.onset,
                        'duration': events_table.duration})
        confound_matrix = make_first_level_design_matrix(
                frame_times,
                error_,
                hrf_model='spm',
                drift_model=None,
                add_regs=motion,
                high_pass = 0) 
        
        # plot_design_matrix(confound_matrix)
        # plt.show()
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
                drift_model=None,
                high_pass = 0)        

        design_matrix = pd.concat([design_matrix,confound_matrix],axis=1)
        
        design_matrix['RightGoL_pmod'] = float('nan')
        design_matrix['RightGoL_pmod'] = right_go_pmodsL['RightGoL']
        design_matrix['LeftGoL_pmod'] = float('nan')
        design_matrix['LeftGoL_pmod'] = left_go_pmodsL['LeftGoL']
        design_matrix['RightGoH_pmod'] = float('nan')
        design_matrix['RightGoH_pmod'] = right_go_pmodsH['RightGoH']
        design_matrix['LeftGoH_pmod'] = float('nan')
        design_matrix['LeftGoH_pmod'] = left_go_pmodsH['LeftGoH']

        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(design_matrix)
        plt.show()
        
        order = ['LeftGoH','LeftGoH_pmod', 'LeftGoL','LeftGoL_pmod',
                 'RightGoH','RightGoH_pmod', 'RightGoL','RightGoL_pmod',
                 'NoGoH','NoGoL','Sad','Neutral','Happy','Win','Loss','Error']
        
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
        
        # drop emotions?
        #design_matrix = design_matrix.drop(columns=['Sad','Neutral','Happy'])

        # put the design matrix in a list
        design_matrices.append(design_matrix)
        
        
        from nilearn import plotting
        dm = plotting.plot_design_matrix(design_matrix)
        fig = dm.get_figure()
        fig.savefig(os.path.join(combination_output,'designs','run-{}_design_matrix.png'.format(idx)))
        
    return design_matrices,design_matrix