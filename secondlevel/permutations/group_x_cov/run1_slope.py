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
                      'derivatives/task_output/SecondLevel_permutations/group_x_slope/{}'.format(run))



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

from nilearn.mass_univariate import permuted_ols
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
from nilearn.maskers import NiftiMasker
grey_matter_masker = NiftiMasker(mask_img=mask)

gm_data = grey_matter_masker.fit_transform(second_level_input)

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
    
    if contrast in ['righthighlow','righthighlowPmod']:
        sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run == 1) &
                       (beh_mri.Mirror != 1) & (beh_mri.GripResponse == 1)]
    elif contrast in ['lefthighlow','lefthighlowPmod']:
        sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run == 1) &
                       (beh_mri.Mirror != 1) & (beh_mri.GripResponse == -1)]
    else:
        sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run == 1) &
                       (beh_mri.Mirror != 1) & (beh_mri.GripResponse != 0)]
    
    if contrast in ['go','nogo','gonogo','goPmod']:
        beh12 = sub_data['Slope'].mean()
    if contrast in ['leftright','leftrightPmod']:
        beh1 = sub_data[sub_data['GripResponse'] == -1]['Slope'].mean()
        beh2 = sub_data[sub_data['GripResponse'] == 1]['Slope'].mean()
        beh12 = beh1-beh2
    if contrast in ['rightleft','rightleftPmod']:
        beh1 = sub_data[sub_data['GripResponse'] == 1]['Slope'].mean()
        beh2 = sub_data[sub_data['GripResponse'] == -1]['Slope'].mean()
        beh12 = beh1-beh2
    if contrast in ['sadneutral']:
        beh12 = sub_data['STAIT'].mean()
    if contrast in ['sadhappy']:
        beh12 = sub_data['STAIT'].mean()
    if contrast in ['happysad']:
        beh12 = sub_data['STAIT'].mean()
    if contrast in ['happyneutral']:
        beh12 = sub_data['STAIT'].mean()
    if contrast in ['winloss']:
        beh1 = sub_data[sub_data['RewardReceived'].isin([5, 20])]['Slope'].mean()
        beh2 = sub_data[sub_data['RewardReceived'].isin([-5, -20])]['Slope'].mean()
        beh12 = beh1-beh2
    if contrast in ['highlow','highlowPmod','righthighlow','righthighlowPmod','lefthighlow','lefthighlowPmod']:
        beh1 = sub_data[sub_data['RewardPromise'] == 1]['Slope'].mean()
        beh2 = sub_data[sub_data['RewardPromise'] == -1]['Slope'].mean()
        beh12 = beh1-beh2
    if contrast in ['lowhigh','lowhighPmod']:
        beh1 = sub_data[sub_data['RewardPromise'] == -1]['Slope'].mean()
        beh2 = sub_data[sub_data['RewardPromise'] == 1]['Slope'].mean()
        beh12 = beh1-beh2


    # ehi_mean = sub_data['EHI'].mean()
    # mdi_mean = sub_data['MDI'].mean()
    # bisbas_bas_mean = sub_data['BISBAS_bas'].mean()
    # age_mean = sub_data['age'].mean()
    # gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,
              'Code':behav_code,
              'Slope':beh12}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)
df_clean = df_clean[~df_clean['SubNr'].isin(exclude_list)]
df_clean=df_clean.iloc[:,2:]

beh_vars = np.array(df_clean['Slope']).reshape(-1, 1)

ols_outputs = permuted_ols(
    beh_vars, 
    gm_data,
    model_intercept=True,
    masker=grey_matter_masker,
    tfce=False,
    n_perm=10000,
    verbose=0,
    n_jobs=10, 
    output_type="dict",
    threshold=0.01
)

t_permuted_ols_unmasked = grey_matter_masker.inverse_transform(
    ols_outputs["t"][0, :]  # select first regressor
)
t_permuted_ols_unmasked.to_filename(os.path.join(cont_output,'{}_t.nii.gz'.format(contrast)))


logp_max_size_unmasked = grey_matter_masker.inverse_transform(
    ols_outputs["logp_max_size"][0, :]  # select first regressor
)
logp_max_size_unmasked.to_filename(os.path.join(cont_output,'{}_logp_max_size.nii.gz'.format(contrast)))

threshold = -np.log10(0.05) # should be 5 % corrected
bwr = plt.cm.bwr

plot1 = plotting.plot_stat_map(
    logp_max_size_unmasked,
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














