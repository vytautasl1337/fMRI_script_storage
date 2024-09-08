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
# 2 streams
# sbatch --array=1-2 Second_Level.sh

run='run_1'

output = os.path.join(working_dir,
                      'derivatives/task_output/SecondLevel_tfce/tfce/{}'.format(run))


map_input = os.path.join(working_dir,
                         'derivatives/task_output/FirstLevel_pmod_0529/{}/{}/first_level_maps')


labels = ['Go','Go>NoGo','NoGo','Left>Right','Right>Left',
          'Sad>Neutral','Sad>Happy','Happy>Sad','Happy>Neutral',
          'Win>Loss','High>Low','Low>High',
          'Go pmod','Left>Right pmod','Right>Left pmod',
          'High>Low pmod','Low>High pmod']
label=labels[index-1]

contrasts = ['go','gonogo','nogo','leftright','rightleft',
             'sadneutral','sadhappy','happysad','happyneutral',
             'winloss','highlow','lowhigh',
             'goPmod','leftrightPmod','rightleftPmod',
             'highlowPmod','lowhighPmod']
contrast=contrasts[index-1]


# exclude participants
exclude_list = ['sub-01','sub-08','sub-036']
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

cont_output = os.path.join(output,contrast)
if not os.path.isdir(cont_output):
    savepath = os.makedirs(cont_output) 
    
second_level_input = []
for sub in range(len(participants_list)):
    subject_id = participants_list.participant_id[sub]
    if subject_id not in exclude_list:
        stat_map = os.path.join(working_dir,
            'derivatives/task_output/FirstLevel_pmod_0529/{}/{}/{}/first_level_maps/{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(subject_id,session,run,subject_id,contrast))
        second_level_input.append(stat_map)
        
n_subjects = len(second_level_input)

design_matrix = pd.DataFrame(
[1] * len(second_level_input),
columns=["intercept"],
)


from nilearn import plotting
dm = plotting.plot_design_matrix(design_matrix)
fig = dm.get_figure()
fig.savefig(os.path.join(cont_output,'{}_design_matrix.png'.format(contrast)))

out_dict = non_parametric_inference(
    second_level_input,
    design_matrix=design_matrix,
    mask=mask,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=True,
    n_jobs=10,
    threshold=0.001,
    tfce = True,
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

plot2 = plotting.plot_stat_map(
    out_dict["logp_max_tfce"],
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg = False,
    title=label,
    cmap=bwr
)
plot2.savefig(os.path.join(cont_output,'{}_tfce.png'.format(contrast)),dpi=300)


matplotlib.pyplot.close()














