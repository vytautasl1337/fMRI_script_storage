import argparse

parser = argparse.ArgumentParser(description='Second level fMRI analysis')
parser.add_argument(
    "--bidsdir",
    "-bdir",
    help="set output width",
    default="/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/",
)

parser.add_argument("--job_index","-ji", type=int, default=1,help='Job index')

args = parser.parse_args()


if args.bidsdir:
    bidsbase_directory = args.bidsdir

index = args.job_index


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
    
participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)

#################################################################################
signif = .05
correction = 'fdr'
session='ses-PRISMA'
# 5 streams
# sbatch --array=1-5 Second_Level.sh
runs = ['run_1','run_2','run_3','run_12','run_123']
run = runs[index-1]

output = os.path.join(working_dir,
                      'derivatives/task_output/SecondLevel_0528_gm/Main_effects/{}_{}/{}'.format(correction,signif,run))



map_input = os.path.join(working_dir,
                         'derivatives/task_output/FirstLevel/{}/{}/first_level_maps')


labels = ['Go High','Go Low','LeftGo High','LeftGo Low','RightGo High','RightGo Low','Go High>Low',
          'Left go','Right go','Left>Right','Right>Left','Go','NoGo','Go>NoGo','High','Low',
          'Happy faces','Sad faces','Neutral faces','Win','Loss','Win>Loss','Loss>Win',
          'Go high','Go low','Sad NoGo','Happy NoGo','Neutral NoGo','Sad>Happy','Sad>Neutral',
          'Happy>Neutral','Sad Go','Happy Go','Neutral Go','High reward > low reward',
          'Low reward > high reward','Sad>Happy']
contrasts = ['gohigh','golow','gohighleft','golowleft','gohighright','golowright','gohighlow',
            'leftgo','rightgo','leftright','rightleft','go','nogo','gonogo','high','low',
             'happy','sad','neutral','win','loss','winloss','losswin',
             'gohigh','golow','sadnogo','happynogo','neutralnogo','sadhappy','sadneutral',
             'happyneutral','sadgo','happygo','neutralgo','highlow','lowhigh','sadhappy']


# exclude participants
exclude_list = ['sub-01','sub-08','sub-036']
#+right handers
#exclude_list = ['sub-01','sub-08','sub-036','sub-012','sub-015','sub-025','sub-030']
#####################################################################################################
if not os.path.isdir(output):
    print('Creating folder for second level task results...')
    savepath = os.makedirs(output) 
    
# Load behavioral data for age, handedness and age
beh_dir = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/'
beh_ref_dir = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/'

beh_results = pd.read_excel(os.path.join(beh_dir,'behavioral_group_output_withreject.xlsx'))
beh_ref = pd.read_excel(os.path.join(beh_ref_dir,'participant_list.xlsx'))
# Organize behavioral data
beh_mri = beh_results[beh_results.group == 1] #MRI data
beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else 0)

data_list=[]
for b in range(beh_ref.shape[0]):

    behav_code = beh_ref.iloc[b,1]
    
    sub_code = participants_list.iloc[b,0]
    
    sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.GripResponse != 0)]
    
    ehi_mean = sub_data['EHI'].mean()
    age_mean = sub_data['age'].mean()
    gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,'Code':behav_code,'age':age_mean,'ehi':ehi_mean,'gender':gender_mean}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)

# find behavioral NaNs, some participants were lazy to fill those in, updates exclude_list
missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['ehi'])].unique()
exclude_list.extend(set(missing_data) - set(exclude_list))
# Now clean matrix
beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
beh=beh.iloc[:,2:]
beh = beh.reset_index(drop=True)

beh_z = beh-beh.mean()

participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)
from nilearn import datasets
bg = datasets.load_mni152_template(resolution=1)
#mask = image.load_img('/mnt/projects/PBCTI/ds-MRI-study/templates/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
mask = datasets.load_mni152_gm_mask()

# Create design and model for each contrast:
for ind,cont in enumerate(contrasts):
    cont_output = os.path.join(output,cont)
    if not os.path.isdir(cont_output):
        savepath = os.makedirs(cont_output) 
    second_level_input = []
    for sub in range(len(participants_list)):
        subject_id = participants_list.participant_id[sub]
        if subject_id not in exclude_list:
            stat_map = os.path.join(working_dir,
                'derivatives/task_output/FirstLevel_0508/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,cont))
            second_level_input.append(stat_map)
            
    n_subjects = len(second_level_input)
    age_ = np.array(beh_z['age']).reshape(-1, 1)
    ehi_ = np.array(beh_z['ehi']).reshape(-1, 1)
    gender_ = np.array(beh_z['gender']).reshape(-1, 1)
    condition_effect = np.hstack(([1] * n_subjects))
    
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis],
                age_,ehi_,gender_)),
        columns=[labels[ind],'age','ehi','gender'],
        )
    
    from nilearn import plotting
    dm = plotting.plot_design_matrix(design_matrix)
    fig = dm.get_figure()
    fig.savefig(os.path.join(cont_output,'{}_design_matrix.png'.format(cont)))
    
    
    second_level_model_ = SecondLevelModel(mask_img=mask)
    second_level_model = second_level_model_.fit(
    second_level_input,
    design_matrix=design_matrix,
    )   

    maps = second_level_model.compute_contrast([1,0,0,0], output_type="all")
    from nilearn.glm import threshold_stats_img
    _, threshold = threshold_stats_img(maps['z_score'], alpha=signif, height_control=correction)
    print('Threshold is ', threshold)


    plot = plotting.plot_stat_map(
        maps['z_score'],
        threshold=threshold,
        colorbar=True,
        display_mode="ortho",
        draw_cross = False,
        black_bg = False,
    )

    plot.savefig(os.path.join(cont_output,'{}.png'.format(cont)),dpi=300)


    inter_view = plotting.view_img(maps['z_score'], threshold=threshold, bg_img = bg)
    inter_view.save_as_html(os.path.join(cont_output,'{}.html'.format(cont)))
    
    from nilearn.reporting import make_glm_report,get_clusters_table
    report = make_glm_report(
            model=second_level_model,
            contrasts=[1,0,0,0],
            height_control = 'fdr',
            alpha = 0.05,
        )
    report.save_as_html(os.path.join(cont_output,'{}_report.html'.format(cont)))
    cluster_table = get_clusters_table(stat_img=maps['z_score'],stat_threshold=threshold)
    cluster_table.to_csv(os.path.join(cont_output,'{}_clusters.csv'.format(cont)))
    
    for i,key in enumerate(maps.keys()):
            maps[key].to_filename(os.path.join(cont_output,'{}_stat-{}_statmap.nii.gz'.format(cont,key)))

    del plot, maps
    matplotlib.pyplot.close()














