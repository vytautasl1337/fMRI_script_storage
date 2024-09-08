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
from nilearn import plotting
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'
    
participants_list = pd.read_csv(os.path.join(working_dir,'participants.tsv'), sep='\t', header = 0)

#################################################################################
signif = .001

corrections_list = ['Unc<{}-'.format(signif)]*12*2


labels_list = ['GoHigh>GoLow',
                                         
                'Go',
                                                                
                'Right high > right low',
                                
                'right go high',
                                
                'right go low',
                                
                'Left high > left low',
                                
                'left go high',
                                
                'left go low',
                                
                'right>left',
                                
                'right',
                                
                'left',
                
                'win>loss']

labels = [label for label in labels_list for _ in range(2)]

cond = labels[index-1]


behaviorals_list = ['Slope']*12

behavioral_ = [behaviorals for behaviorals in behaviorals_list for _ in range(2)]


contrasts = ['[1,0,0,0]','[0,1,0,0]']*12


contrast = ast.literal_eval(contrasts[index-1])  #42 subjects


label = corrections_list[index-1]+labels[index-1]+'.'+behavioral_[index-1]+'.'+contrasts[index-1]


contrast_list_ = ['gohighgolow',
                  'go',
                  'righthighrightlow',
                  'rightgohigh',
                  'rightgolow',
                  'lefthighleftlow',
                  'leftgohigh',
                  'leftgolow',
                  'rightleft',
                  'rightgo',
                  'leftgo',
                  'winloss']




contrast_list = [contr for contr in contrast_list_ for _ in range(2)]
#contrast2_list = [contr for contr in contrast2_list_ for _ in range(2)]

contrast1 = contrast_list[index-1]
#contrast2 = contrast2_list[index-1]

behavioral = behavioral_[index-1]



# exclude participants
#exclude_list = ['sub-08','sub-036']
#righters
exclude_list = ['sub-08','sub-036','sub-012','sub-015','sub-025','sub-030']
#####################################################################################################
save_folder = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/SecondLevel/SecondLevel_1and2_run_03hz/{}/{}'.format(signif,label)
if not os.path.isdir(save_folder):
    print('Creating folder for task results')
    savepath = os.makedirs(save_folder) 
    


from design_control import design
second_level_input,design_matrix,n_subjects=design(contrast1,contrast,
                                                save_folder,exclude_list,participants_list,
                                                working_dir,label,behavioral,cond)




from nilearn import datasets
bg = datasets.load_mni152_template(resolution=1)
mask = datasets.load_mni152_brain_mask()

#second_level_model = SecondLevelModel()
second_level_model_ = SecondLevelModel(mask_img=mask)
second_level_model = second_level_model_.fit(
    second_level_input,
    design_matrix=design_matrix,
)

z_map = second_level_model.compute_contrast(contrast, output_type="z_score")
# from nilearn.glm import threshold_stats_img
# _, threshold = threshold_stats_img(z_map, alpha=signif, height_control='fdr')

import scipy
threshold=scipy.stats.norm.isf(signif)

plot = plotting.plot_stat_map(
    z_map,
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg = False,
    #title=label,
)
#plotting.show()

plot.savefig(os.path.join(save_folder,'{}.png'.format(label)),dpi=300)


inter_view = plotting.view_img(z_map, threshold=threshold, bg_img = bg, title = label)
#inter_view.open_in_browser()
inter_view.save_as_html(os.path.join(save_folder,'{}.html'.format(label)))














