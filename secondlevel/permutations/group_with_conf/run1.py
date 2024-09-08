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
#################################################################################
session='ses-PRISMA'

run='run_1'

output = os.path.join(working_dir,
                      'derivatives/task_output/SecondLevel_permutations/group_x_conf/{}'.format(run))


labels = ['Go','Go>NoGo','NoGo','Left>Right','Right>Left',
          'Sad>Neutral','Sad>Happy','Happy>Neutral',
          'Win>Loss','High>Low','Low>High','Righ high>low','Left high>low',
          'Go pmod','Left>Right pmod','Right>Left pmod',
          'High>Low pmod','Low>High pmod','Right High>Low pmod','Left High>Low pmod']
label=labels[index-1]

contrasts = ['go','gonogo','nogo','leftright','rightleft',
             'sadneutral','sadhappy','happyneutral',
             'winloss','highlow','lowhigh','righthighlow','lefthighlow',
             'goPmod','leftrightPmod','rightleftPmod',
             'highlowPmod','lowhighPmod','righthighlowPmod','lefthighlowPmod']
contrast=contrasts[index-1]


# exclude participants
exclude_list = ['sub-08','sub-036']
#+right handers
#exclude_list = ['sub-01','sub-08','sub-036','sub-012','sub-015','sub-025','sub-030']
#####################################################################################################
if not os.path.isdir(output):
    savepath = os.makedirs(output) 
    

participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)
from nilearn import datasets
bg = datasets.load_mni152_template(resolution=1)
mask = datasets.load_mni152_gm_mask()

from nilearn.glm.second_level import non_parametric_inference
# Create design and model for each contrast:

cont_output = os.path.join(output,contrast)
if not os.path.isdir(cont_output):
    savepath = os.makedirs(cont_output) 
    
second_level_input = []
for sub in range(len(participants_list)):
    subject_id = participants_list.participant_id[sub]
    if subject_id not in exclude_list:
        stat_map = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0605/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrast))
        second_level_input.append(stat_map)
        
n_subjects = len(second_level_input)

beh_dir = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/'
beh_ref_dir = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/'
beh_results = pd.read_excel(os.path.join(beh_dir,'behavioral_group_output_withreject.xlsx'))
beh_ref = pd.read_excel(os.path.join(beh_ref_dir,'participant_list.xlsx'))

beh_mri = beh_results[beh_results.group == 1] #MRI data
beh_mri.loc[:, 'gender'] = beh_mri['gender'].apply(lambda x: 1 if x == 'Female' else 0)

data_list=[]

for b in range(beh_ref.shape[0]):

    behav_code = beh_ref.iloc[b,1]
    
    sub_code = participants_list.iloc[b,0]
    

    sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run == 1) &
                    (beh_mri.Mirror != 1) & (beh_mri.GripResponse != 0)]

    ehi_mean = sub_data['EHI'].mean()
    # mdi_mean = sub_data['MDI'].mean()
    # bisbas_bas_mean = sub_data['BISBAS_bas'].mean()
    age_mean = sub_data['age'].mean()
    gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,
              'Code':behav_code,
              'ehi':ehi_mean,
              'age':age_mean,
              'gender':gender_mean}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)
df_clean = df_clean[~df_clean['SubNr'].isin(exclude_list)]
df_clean=df_clean.iloc[:,2:]

age_ = np.array(df_clean['age']).reshape(-1, 1)
ehi_ = np.array(df_clean['ehi']).reshape(-1, 1)
gender_ = np.array(df_clean['gender']).reshape(-1, 1)
condition_effect = np.hstack(([1] * n_subjects))
    
design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis],
            age_,ehi_,gender_)),
    columns=['intercept','age','ehi','gender'],
    )


from nilearn import plotting
dm = plotting.plot_design_matrix(design_matrix)
fig = dm.get_figure()
fig.savefig(os.path.join(cont_output,'{}_design_matrix.png'.format(contrast)))

out_dict = non_parametric_inference(
    second_level_input,
    second_level_contrast='intercept',
    design_matrix=design_matrix,
    mask=mask,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=True,
    n_jobs=10,
    threshold=0.001,
    tfce = False,
)

for i,key in enumerate(out_dict.keys()):
    out_dict[key].to_filename(os.path.join(cont_output,'{}_{}.nii.gz'.format(contrast,key)))
    

threshold = -np.log10(0.05) # should be 5 % corrected
bwr = plt.cm.bwr

plot1 = plotting.plot_stat_map(
    out_dict["logp_max_size"],
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    title=label,
    black_bg=False,
    cmap=bwr
)

plot1.savefig(os.path.join(cont_output,'{}_clustersize.png'.format(contrast)),dpi=300)

matplotlib.pyplot.close()













