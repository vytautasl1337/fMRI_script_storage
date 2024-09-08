import pandas as pd
from scipy.io import loadmat
import os,numpy,re
from nilearn.glm.first_level import make_first_level_design_matrix

import matplotlib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_design_matrix

def design_mat(fmri_task,beh_table,events_dir,physio_dir,comb,subject_id,task_label,t_r,
               combination_output,beh_value,phys_exclude,confound_dir,working_dir,session):
    
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
        # REWARD VS PUNISHMENT
        events_table['reward_feedback'].replace({-5: 'Loss', -20: 'Loss', 5: 'Win', 20: 'Win'}, inplace=True)
        ######################################################################################################
        # Add behavioral value
        events_table[beh_value] = subject_beh[beh_value].values
        ######################################################################################################
        # SEPARATE GO/NOGO RESPONSES
        go_conditions = []
        for g,row in enumerate(events_table['response_type']):
            if   row == -1:   
                go_conditions.append('Left')
            elif row ==  1:   
                go_conditions.append('Right')
            elif row ==  0:   
                go_conditions.append('NoGo')
                
        events_table['trial']=go_conditions
        
        ######################################################################################################
        # REPLACE REWARD CONDITIONS WITH 1 AND 0
        events_table['trial_type'] = events_table['trial_type'].apply(lambda x: 1 if x.startswith('H') else 0)
        ######################################################################################################
        #ERRORS
        errors=[]    
        for g,row in enumerate(events_table['correct_trial']):
            if   row == 0:   errors.append('Correct')
            elif row == 1:   errors.append('Error')
        events_table['correct_trial']=errors
        
        # SEPARATE LEFT AND RIGHT AND NOGO
        go_events = events_table[(events_table['trial']!='NoGo') & (events_table['correct_trial']=='Correct')]
        go_events['trial'] = go_events['trial'].apply(lambda x: 'Go' if x=='Left' else 'Go')
        
        left_events = events_table[(events_table['trial']=='Left') & (events_table['correct_trial']=='Correct')]
        right_events = events_table[(events_table['trial']=='Right') & (events_table['correct_trial']=='Correct')]
        nogo_events = events_table[(events_table['trial']=='NoGo') & (events_table['correct_trial']=='Correct')]
        
        
        # 0 CENTER SLOPES
        mean_left_slope = numpy.mean(left_events[beh_value])
        left_events[beh_value] = left_events[beh_value] - mean_left_slope
        
        mean_right_slope = numpy.mean(right_events[beh_value])
        right_events[beh_value] = right_events[beh_value] - mean_right_slope
        
        # 0 CENTER REWARD CONDITION (NEEDED?)
        mean_condition = numpy.mean(go_events['trial_type'])
        go_events['trial_type'] = go_events['trial_type'] - mean_condition
        

        # DEFINE VOLUME ONSETS
        frame_times = numpy.arange(n_scans) * t_r
        ######################################################################################################
        # Get pmods of left behavior 
        events_pmod = pd.DataFrame({'trial_type': left_events['trial'], 'onset': left_events['onset'],
                        'duration': left_events['duration'], 'modulation': left_events[beh_value]})
        
        design_matrix_pmod_left = make_first_level_design_matrix(
                frame_times,
                events_pmod,
                hrf_model='spm',
                drift_model=None) 
        design_matrix_pmod_left = design_matrix_pmod_left.rename(columns={'Left': 'Left_slope_pmod'})
        
        plot_design_matrix(design_matrix_pmod_left)
        plt.show()
        
        ###
        
        events_pmod2 = pd.DataFrame({'trial_type': right_events['trial'], 'onset': right_events['onset'],
                        'duration': right_events['duration'], 'modulation': right_events[beh_value]})
        
        design_matrix_pmod_right = make_first_level_design_matrix(
                frame_times,
                events_pmod2,
                hrf_model='spm',
                drift_model=None) 
        
        design_matrix_pmod_right = design_matrix_pmod_right.rename(columns={'Right': 'Right_slope_pmod'})
        
        plot_design_matrix(design_matrix_pmod_right)
        plt.show()
        
        ###
        
        events_pmod_rew = pd.DataFrame({'trial_type': go_events['trial'], 'onset': go_events['onset'],
                'duration': go_events['duration'], 'modulation': go_events['trial_type']})
        
        design_matrix_pmod_rew = make_first_level_design_matrix(
                frame_times,
                events_pmod_rew,
                hrf_model='spm',
                drift_model=None) 
        
        design_matrix_pmod_rew = design_matrix_pmod_rew.rename(columns={'Go': 'Reward_prospect_pmod'})
        
        plot_design_matrix(design_matrix_pmod_rew)
        plt.show()
        
        ######################################################################################################
        # BUILD CONFOUNDS
        confound_table = pd.read_csv(confound_dir.format(subject_id,session,subject_id,session,task_label,idx), sep='\t')
        
        cosines = confound_table.filter(like='cosine')
        
        fd_mp = confound_table[['framewise_displacement',
                                    'trans_x','trans_y','trans_z',
                                    'rot_x','rot_y','rot_z']].copy()
        
        confounds = pd.concat([fd_mp,cosines],axis=1)
        
        confounds = confounds - confounds.mean()
        
        # Add respiratory/pulse data and merge all confounds and errors
        if subject_id not in phys_exclude:
            physio_file = loadmat(physio_dir.format(subject_id,subject_id,session,task_label,idx))
            data_physio = [[row.flat[0] for row in line] for line in physio_file['R']]
            motion_phys = pd.DataFrame(data_physio)
            motion = pd.concat([motion_phys,confounds],axis=1)
        else:
            motion = confounds
        motion.fillna(0,inplace=True)
        
        pmod_regressors = pd.concat([design_matrix_pmod_left['Left_slope_pmod'],
                                     design_matrix_pmod_right['Right_slope_pmod'],
                                     design_matrix_pmod_rew['Reward_prospect_pmod']],axis=1)
        
        plot_design_matrix(pmod_regressors)
        plt.show()
        

        
        error_ = pd.DataFrame({'trial_type': events_table.correct_trial, 'onset': events_table.onset,
                        'duration': events_table.duration})
        confound_matrix = make_first_level_design_matrix(
                frame_times,
                error_,
                hrf_model='spm',
                drift_model=None,
                add_regs=motion) 
        
        confound_matrix = confound_matrix.drop(columns=['constant', 'Correct'])
        # If no errors occured, create empty Error column
        if 'Error' not in confound_matrix.columns:
            confound_matrix['Error'] = [0]*n_scans
            col_index = confound_matrix.columns.get_loc('Error')
            confound_matrix.insert(0, 'Error', confound_matrix.pop('Error'))
        
        plot_design_matrix(confound_matrix)
        plt.show()
        
        
        
        ######################################################################################################
        # START BUILDING DESIGN MATRIX
        conditions_ = pd.concat([events_table.reward_feedback,
                                 left_events.trial,
                                 right_events.trial,
                                 nogo_events.trial
                                 ],axis=0)
        
        duration_ = pd.concat([events_table.reward_duration,
                               left_events.duration,
                               right_events.duration,
                               nogo_events.duration
                               ],axis=0)
        
        onsets_ = pd.concat([events_table.reward_onset,
                             left_events.onset,
                             right_events.onset,
                             nogo_events.onset,
                             ],axis=0)
        
        events = pd.DataFrame({'trial_type': conditions_, 'onset': onsets_,
                        'duration': duration_})
        
        confs = pd.concat([pmod_regressors,confound_matrix],axis=1)
        
        design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model='spm',
                drift_model=None,
                high_pass = 0,
                add_regs=confs)   
        
        plot_design_matrix(design_matrix)
        plt.show()
        
        # Arrange

        order = ['Left','Left_slope_pmod','Right','Right_slope_pmod','Reward_prospect_pmod',
                 'NoGo','Win','Loss','Error']
        
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