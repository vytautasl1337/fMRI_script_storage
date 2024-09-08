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
signif = .01
correction = 'fpr'
runs = ['run_1','run_123']
exclude_list = ['sub-01','sub-08','sub-036']
contrast1=['gohigh','gohighleft','gohighright']
contrast2=['golow','golowleft','golowright']
TITLE = ['Go High > Go Low * Slope','Go High Left > Go Low Left * Slope', 'Go High Right > Go Low Right * Slope']

# Loop over runs
for run_nr,run in enumerate(runs):
    
    for contrast in range(3):

        images_dir = os.path.join(working_dir,'derivatives/task_output/FirstLevel_pmod/{}/ses-PRISMA',
                                run,'first_level_maps')

        maps1,maps2=[],[]
        for sub in participants_list.participant_id:
            if sub not in exclude_list:
                stat_map = os.path.join(images_dir.format(sub),
                                        '{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(sub,contrast1[contrast]))
                maps1.append(stat_map)
                stat_map = os.path.join(images_dir.format(sub),
                                        '{}_contrast-{}_stat-effect_size_statmap.nii.gz'.format(sub,contrast2[contrast]))
                maps2.append(stat_map)
         
        maps = maps1+maps2
        
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
        
        condition_effect = np.hstack(([1] * n_subjects))


        design_matrix = pd.DataFrame(
                    np.hstack((condition_effect[:, np.newaxis])),
                    columns=["Slope*BOLD"],
                                                    )



        from nilearn import datasets
        bg = datasets.load_mni152_template(resolution=1)
        mask = datasets.load_mni152_gm_mask()

        second_level_model_ = SecondLevelModel(mask_img=mask).fit(
            maps, design_matrix=design_matrix
        )

        stat_maps_ = second_level_model_.compute_contrast(
            "Slope*BOLD", output_type="all"
        )

        from nilearn.glm import threshold_stats_img
        _, threshold = threshold_stats_img(stat_maps_['z_score'], 
                                        alpha=signif, 
                                        height_control=correction)
        print('Threshold is ', threshold)
        title_str = TITLE[contrast] + '_' + str(correction) + '<' + str(signif) + '_' + runs[run_nr]
        
        display = plotting.plot_stat_map(
            stat_maps_['z_score'],
            threshold=threshold,
            colorbar=True,
            display_mode="mosaic",
            draw_cross = False,
            black_bg = False,
            title = title_str
        )
        #plt.show()
        display.savefig('pics/{}.png'.format(title_str),dpi=400)

        inter_view = plotting.view_img(stat_maps_['z_score'], threshold=threshold, bg_img = bg, title = title_str)
        inter_view.save_as_html('pics/{}.html'.format(title_str))



















