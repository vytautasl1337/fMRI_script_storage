#######################################################################################################
#######################################################################################################
import argparse

parser = argparse.ArgumentParser(description='First level fMRI analysis')
parser.add_argument(
    "--bidsdir",
    "-bdir",
    help="set output width",
    default="/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/",
)

parser.add_argument(
    "--subid", "-s", help="Subject identifier", default="sub-01"
)
args = parser.parse_args()

if args.bidsdir:
    bidsbase_directory = args.bidsdir
if args.subid:
    subject_id = str(args.subid)
#######################################################################################################
#IMPORT LIBRARIES
from tkinter import *
import pandas as pd
import os,re
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
from nilearn.image import load_img
from nilearn.masking import compute_multi_background_mask
import matplotlib,itertools
### Load functions
from first_level_func.concat_func import concat_func_runs
from first_level_func.design import design_mat
from first_level_func.contrasts import my_contrasts

# CHANGE
working_dir = bidsbase_directory#'/mnt/scratch/PBCTI/BIDS_FINAL/BIDS_dataset_20230116-1320/'
#working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'
########################
#print(participants_list)
######################## Directories and settings
fmriprep_dir = os.path.join(working_dir,'derivatives/fmriprep-22.0.2') # path to fmriprep outputs
events_dir = os.path.join(working_dir,'raw/{}/{}/func/','{}_{}_task-{}_run-{}_events.tsv') # events file directory
confound_dir = os.path.join(fmriprep_dir,'{}/{}/func/{}_{}_task-{}_dir-ap_run-{}_part-mag_desc-confounds_timeseries.tsv') # confound file directory
physio_dir = os.path.join(working_dir,'derivatives/phys/{}/{}_{}_task-{}_run-{}_phys.mat')
task_file = os.path.join(fmriprep_dir,'{}/{}/func/{}_{}_task-{}_dir-ap_run-{}_part-mag_space-{}_desc-preproc_bold.nii.gz')
derivatives_folder = os.path.join(os.path.join(working_dir,'derivatives/task_output/FirstLevel/FirstLevel_pmods_new'))

beh_value = 'Slope'
t_r = 1.88 # TR in seconds
session = 'ses-PRISMA' # session name
task_label = 'ROSETTA' # task name
space_label = 'MNI152NLin2009cAsym_res-2' # anatomical space
smooth_mm = 6 # smoothing parameter for the model
phys_exclude = ['sub-011','sub-017'] # if you have subjects with inconsitant phys measures, type in to exclude, otherwise type something unrealistic

# Behavioral data sheet for error inclusion
beh_path = '/mnt/projects/PBCTI/combined/Results/Group_output_202406131123/'
beh_table = pd.read_excel(os.path.join(beh_path,'behavioral_group_output_withreject.xlsx'))

################################################################################
# Make derivative folder
if not os.path.isdir(derivatives_folder):
    print('Creating folder for task results')
    savepath = os.makedirs(derivatives_folder)  


           
print('Analyzing participant {}'.format(subject_id))
subject_folder = os.path.join(fmriprep_dir,subject_id,'{}/func'.format(session))
subject_folder_anat = os.path.join(fmriprep_dir,subject_id,'{}/anat'.format(session))
subject_output = os.path.join(derivatives_folder,subject_id,session)

# Move anatomical and functional data ############################################
import shutil,os,glob
if not os.path.isdir(subject_output):
    subpath = os.makedirs(subject_output)
    
# Expecting single anatomical file
shutil.copyfile(os.path.join(subject_folder_anat,'{}_ses-PRISMA_space-{}_desc-preproc_T1w.nii.gz'.format(subject_id,space_label)),
                        os.path.join(subject_output,'{}_ses-PRISMA_space-{}_desc-preproc_T1w.nii.gz'.format(subject_id,space_label)))

# Figure out how many scans the subject has, it may vary for each subject
funcs_in_folder = []
for root,dirs,files in os.walk(subject_folder):
    for names in files:
        if names.endswith('desc-preproc_bold.nii.gz') and task_label in names:
            print(names,' found')
            funcs_in_folder.extend(glob.glob(os.path.join(root,names)))
# Sort
funcs_in_folder.sort()

# Get functional run numbers
task_runs = []
for filename in funcs_in_folder:
    match = re.search(r'run-(\d+)', filename)
    task_runs.append(int(match.group(1)))
        
# Move those functional runs to subject_output folder
for run_id in task_runs:
    nifti_name = '{}_{}_task-{}_dir-ap_run-{}_part-mag_space-{}_desc-preproc_bold.nii.gz'.format(subject_id,
                                                                                                 session,
                                                                                                 task_label,
                                                                                                 run_id,
                                                                                                 space_label)
    nifti_data = os.path.join(subject_folder,nifti_name)
    shutil.copyfile(nifti_data,
                        os.path.join(subject_output,nifti_name.format(subject_id,session,task_label,run_id+1,space_label)))

anat_image = os.path.join(subject_output,'{}_{}_space-{}_desc-preproc_T1w.nii.gz'.format(subject_id,session,space_label))


# Concat all functional task runs into a list (1,2,3,1+2,1+2+3) - more runs, more work, and analyze each pair
# Make possible combinations, might be handy to look at separate runs or if the first run was done before treatment or similar
combinations = []
for r in range(1, len(task_runs) + 1):
    combinations.extend(list(itertools.combinations(task_runs, r)))

combinations = [list(comb) for comb in combinations]


# Run analysis on each combination
nifti_name = '{}_{}_task-{}_dir-ap_run-{}_part-mag_space-{}_desc-preproc_bold.nii.gz'

for index,comb in enumerate(combinations):
    print(comb)
    fmri_task = concat_func_runs(combinations,index,subject_output,nifti_name,
                     subject_id,session,space_label,task_label)
    
    combination_output = os.path.join(subject_output,'run_'+''.join(map(str, comb)))
    if not os.path.isdir(combination_output):
        subpath = os.makedirs(combination_output)

    # Construct design matrix
    print('Building the design matrix for run(s)', comb)
    

    design_matrices,design_matrix=design_mat(fmri_task,beh_table,events_dir,physio_dir,comb,
                                        subject_id,task_label,t_r,combination_output,beh_value,
                                        phys_exclude,confound_dir,working_dir,session)

    # Specify basic contrasts
    print('Specifiying contrasts')
    basic_contrasts,contrasts=my_contrasts(design_matrix)

    # Get common mask from fmriprep
    mask_list=[]
    for run_msk in comb:
        mask_list.append(load_img(os.path.join(fmriprep_dir,
            '{}/{}/func/{}_{}_task-{}_dir-ap_run-{}_part-mag_space-{}_desc-brain_mask.nii.gz'.format(subject_id,
                                                                                                                 session,
                                                                                                                 subject_id,
                                                                                                                 session,
                                                                                                                 task_label,
                                                                                                                 run_msk,
                                                                                                                 space_label))))
    mask = compute_multi_background_mask(mask_list,threshold=1)

    # Fit model
    print('Fitting GLM...')
    fmri_glm = FirstLevelModel(slice_time_ref = .5, 
                               t_r=1.88,
                               hrf_model='spm',
                                minimize_memory=False,
                                standardize=True,
                                smoothing_fwhm=smooth_mm,
                                mask_img=mask,
                                high_pass = 0)

    fmri_glm = fmri_glm.fit(fmri_task, design_matrices=design_matrices)
    del fmri_task

    # Compute the contrasts
    pictures_folder = os.path.join(os.path.join(combination_output,'subject_contrasts_pictures'))
    if not os.path.isdir(pictures_folder):
        print('Creating folder for pictures')
        savepath1 = os.makedirs(pictures_folder)  
        
    first_level_folder = os.path.join(os.path.join(combination_output,'first_level_maps'))   
    if not os.path.isdir(first_level_folder):
        print('Creating folder for first level results')
        savepath2 = os.makedirs(first_level_folder) 
        
        
    from nilearn.glm import threshold_stats_img  
    from nilearn.interfaces.bids.glm import _clean_contrast_name
    from nilearn.plotting import plot_contrast_matrix

    for contrast_id, contrast_val in contrasts.items():
        
        cont = plot_contrast_matrix(contrast_val, design_matrix=design_matrix)
        fig = cont.get_figure()
        fig.savefig(os.path.join(first_level_folder,'{}.png'.format(contrast_id)))

        print('Contrast ', contrast_id)
        print(contrast_val)
        try:
            maps = fmri_glm.compute_contrast(contrast_val, output_type='all')
        except ValueError as e:
            # This is a leftover from aCompCor version, with varying run lengths
            print(e)
            match = re.search(r'length P=(\d+)', str(e))
            expected_length = int(match.group(1))
            print('Predicted length ',expected_length)
            contrast_val = contrast_val[:15]
            maps = fmri_glm.compute_contrast(contrast_val, output_type='all')
        
        correction = 'Uncorrected, p<.001'
        
        _, threshold = threshold_stats_img(maps['z_score'], alpha=0.001, height_control="fpr")
        
        display=plotting.plot_stat_map(
            maps['z_score'], bg_img=anat_image, threshold=threshold , display_mode='ortho',
            black_bg=True, title='{}. {}. {}'.format(subject_id,contrast_id,correction))
        
        display.savefig(os.path.join(pictures_folder,'{}_first_level_{}.png'.format(subject_id,contrast_id)))
        
        contrast_name_for_maps = _clean_contrast_name(contrast_id)
        
        for i,key in enumerate(maps.keys()):
            maps[key].to_filename(os.path.join(first_level_folder,'{}_contrast-{}_stat-{}_statmap.nii.gz'.format(subject_id,
                                                                                                                contrast_name_for_maps,
                                                                                                                key)))
        
        del display, maps
        matplotlib.pyplot.close()
    
    print('Finished run(s)', comb)
print('Finished with participant {}'.format(subject_id))
