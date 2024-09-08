#######################################################################################################
#IMPORT LIBRARIES
import pandas as pd
import os,ast,glob,shutil
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
signif = .001
correction = 'fpr'
run = 'run_1'
exclude_list = ['sub-08','sub-036']
contrast1='gohigh'
contrast2='golow'

images_dir = os.path.join(working_dir,'derivatives/task_output/FirstLevel_pmod/{}/ses-PRISMA',
                          run,'first_level_maps')

# Substract image maps
maps1,maps2=[],[]
for i in participants_list.participant_id:
    if i not in exclude_list:
        stat_map = os.path.join(images_dir.format(i),
                                '{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(i,contrast1))
        maps1.append(stat_map)
        stat_map = os.path.join(images_dir.format(i),
                                '{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(i,contrast2))
        maps2.append(stat_map)
        
import nibabel as nib
# clean folder just in case
for filename in os.listdir('images'):
    file_path = os.path.join('images', filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('No file exists')
        
for i,(map1_file, map2_file) in enumerate(zip(maps1, maps2)):
    map1_img = nib.load(os.path.join(maps1[i]))
    map2_img = nib.load(os.path.join(maps2[i]))
    
    map1_data = map1_img.get_fdata()
    map2_data = map2_img.get_fdata()
    
    subtracted_data = map1_data - map2_data
    subtracted_img = nib.Nifti1Image(subtracted_data, affine=map1_img.affine)
    output_filename = f'sub{i}_subtracted.nii.gz'
    output_path = os.path.join('images', output_filename)
    nib.save(subtracted_img, output_path)

maps = []
for root, dirs, files in os.walk('images'):
    for file in files:
        maps.append(os.path.join(root, file))

n_subjects = len(maps)

# Get additional behavior data
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
    
    sub_data = beh_mri[(beh_mri.id == behav_code) & (beh_mri.Correct == 1) & (beh_mri.Run==1)&
                       (beh_mri.Mirror != 1) & (beh_mri.GripResponse == -1)]

    
    ehi_mean = sub_data['EHI'].mean()
    mdi_mean = sub_data['MDI'].mean()
    bisbas_bas_mean = sub_data['BISBAS_bas'].mean()
    bisbas_bis_mean = sub_data['BISBAS_bis'].mean()
    age_mean = sub_data['age'].mean()
    gender_mean = sub_data['gender'].mean()

    dictio = {'SubNr':sub_code,
              'Code':behav_code,
              'age':age_mean,
              'mdi':mdi_mean,
              'bisbas_bas':bisbas_bas_mean,
              'bisbas_bis':bisbas_bis_mean,
              'gender':gender_mean}
        
    data_list.append(dictio)

df_clean = pd.DataFrame(data_list)

# find behavioral NaNs, some participants were lazy to fill those in, updates exclude_list
missing_data = df_clean['SubNr'].loc[pd.isna(df_clean['age'])].unique()
exclude_list.extend(set(missing_data) - set(exclude_list))
# Now clean matrix
beh = df_clean.loc[~df_clean['SubNr'].isin(exclude_list)]
beh.reset_index(drop=True, inplace=True)
beh['fd'] = np.nan
for index,sub in enumerate(beh.SubNr):
    fd_file = pd.read_csv(os.path.join(working_dir, f"derivatives/fmriprep-22.0.2/{sub}/ses-PRISMA/func/{sub}_ses-PRISMA_task-ROSETTA_dir-ap_run-1_part-mag_desc-confounds_timeseries.tsv"),sep='\t')
    beh.loc[index, 'fd'] = fd_file['framewise_displacement'].mean()

beh=beh.iloc[:,2:]
beh = beh.reset_index(drop=True)
beh_z = beh-beh.mean()

condition_effect = np.hstack(([1] * n_subjects))
age_ = np.array(beh_z['age']).reshape(-1, 1)
gender_ = np.array(beh_z['gender']).reshape(-1, 1)
mdi_ = np.array(beh_z['mdi']).reshape(-1, 1)
bisbas_bas_ = np.array(beh_z['bisbas_bas']).reshape(-1, 1)
bisbas_bis_ = np.array(beh_z['bisbas_bis']).reshape(-1, 1)
fd_ = np.array(beh_z['fd']).reshape(-1, 1)

# design_matrix = pd.DataFrame(
#     np.hstack((condition_effect[:, np.newaxis],
#                age_,
#                gender_,
#                #mdi_,
#                bisbas_bas_,
#                #bisbas_bis_,
               
#                )),
#     columns=['Slope_pmod',
#              'Age',
#              'Gender',
#              #'MDI',
#              'BISBAS_BAS',
#              #'BISBAS_BIS',
#              ],
# )

design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis])),
    columns=['Slope_pmod'],
)

# plot_design_matrix(design_matrix)
# plt.show()
# plt.close()

from nilearn import datasets
from nilearn.image import load_img
bg = datasets.load_mni152_template(resolution=1)
#mask = load_img('/mnt/projects/PBCTI/ds-MRI-study/templates/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz')
mask = datasets.load_mni152_gm_mask()

second_level_model_ = SecondLevelModel(mask_img=mask).fit(
    maps, design_matrix=design_matrix
)

stat_maps_ = second_level_model_.compute_contrast(
    "Slope_pmod", output_type="all"
)



from nilearn.glm import threshold_stats_img
_, threshold = threshold_stats_img(stat_maps_['z_score'], 
                                   alpha=signif, 
                                   #cluster_threshold=10,
                                   height_control=correction)
print('Threshold is ', threshold)

plotting.plot_stat_map(
    stat_maps_['z_score'],
    threshold=threshold,
    colorbar=True,
    display_mode="mosaic",
    draw_cross = False,
    black_bg = False,
)
plt.show()

inter_view = plotting.view_img(stat_maps_['z_score'], threshold=threshold, bg_img = bg)
inter_view.open_in_browser()



















