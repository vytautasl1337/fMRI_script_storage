import nilearn
import os
import matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting import plot_stat_map, plot_roi
from nilearn.image import load_img, resample_to_img, math_img,new_img_like

image_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/FirstLevel_slope'

runs = ['run_1', 'run_2', 'run_3']
exclude = [1, 8, 36]

beh_path = '/mnt/projects/PBCTI/combined/Results/Group_output_202402081541/behavioral_group_output_withreject.xlsx'
beh_file = pd.read_excel(beh_path)

beh_ref_path = '/mnt/projects/PBCTI/ds-MRI-study/MRI_data/participant_list.xlsx'
beh_ref = pd.read_excel(beh_ref_path)

# Mask from atlas
from nilearn.datasets import fetch_atlas_harvard_oxford
ho_cortical = fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
ho_subcortical = fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

regions_of_interest = {
    "Accumbens_L": 11,
    "Accumbens_R": 21,
    "Putamen_L": 6,
    "Putamen_R": 17,
    "Caudate_L": 5,
    "Caudate_R": 16
}

def create_region_mask(region_label, atlas_img):
    return math_img(f"img == {region_label}", img=atlas_img)

accumbens_L_mask = create_region_mask(regions_of_interest["Accumbens_L"], ho_subcortical.maps)
accumbens_R_mask = create_region_mask(regions_of_interest["Accumbens_R"], ho_subcortical.maps)
putamen_L_mask = create_region_mask(regions_of_interest["Putamen_L"], ho_subcortical.maps)
putamen_R_mask = create_region_mask(regions_of_interest["Putamen_R"], ho_subcortical.maps)
caudate_L_mask = create_region_mask(regions_of_interest["Caudate_L"], ho_subcortical.maps)
caudate_R_mask = create_region_mask(regions_of_interest["Caudate_R"], ho_subcortical.maps)

#mPFC_mask = create_region_mask(regions_of_interest["mPFC"], ho_cortical.maps)


# Plot the masks
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
plot_roi(accumbens_L_mask, title="Accumbens_L", axes=axes[0,0], display_mode='ortho')
plot_roi(accumbens_R_mask, title="Accumbens_R", axes=axes[0,1], display_mode='ortho')
plot_roi(putamen_L_mask, title="Putamen_L", axes=axes[0,2], display_mode='ortho')
plot_roi(putamen_R_mask, title="Putamen_R", axes=axes[1,0], display_mode='ortho')
plot_roi(caudate_L_mask, title="Caudate_L", axes=axes[1,1], display_mode='ortho')
plot_roi(caudate_R_mask, title="Caudate_R", axes=axes[1,2], display_mode='ortho')
#plot_roi(mPFC_mask, title="mPFC", axes=axes[2], display_mode='ortho')
plt.show()

from nilearn.maskers import NiftiMasker
refimg = load_img(os.path.join(image_dir,
                               f'sub-01/ses-PRISMA/run_1/first_level_maps/sub-01_contrast-gohighgolow_stat-effect_size_statmap.nii.gz'))

# Resample
resampled_accumbens_L_mask = resample_to_img(accumbens_L_mask, refimg, interpolation='nearest')
resampled_accumbens_R_mask = resample_to_img(accumbens_R_mask, refimg, interpolation='nearest')
resampled_putamen_L_mask = resample_to_img(putamen_L_mask, refimg, interpolation='nearest')
resampled_putamen_R_mask = resample_to_img(putamen_R_mask, refimg, interpolation='nearest')
resampled_caudate_L_mask = resample_to_img(caudate_L_mask, refimg, interpolation='nearest')
resampled_caudate_R_mask = resample_to_img(caudate_R_mask, refimg, interpolation='nearest')

# Set up maskers
masker_accumbens_L_mask = NiftiMasker(mask_img=resampled_accumbens_L_mask)
masker_accumbens_R_mask = NiftiMasker(mask_img=resampled_accumbens_R_mask)
masker_putamen_L_mask = NiftiMasker(mask_img=resampled_putamen_L_mask)
masker_putamen_R_mask = NiftiMasker(mask_img=resampled_putamen_R_mask)
masker_caudate_L_mask = NiftiMasker(mask_img=resampled_caudate_L_mask)
masker_caudate_R_mask = NiftiMasker(mask_img=resampled_caudate_R_mask)

# Get data for each run
beh_ref = beh_ref[~beh_ref['id'].isin(exclude)]
beh_file.loc[:, 'gender'] = beh_file['gender'].apply(lambda x: 1 if x == 'Female' else 0)

Run,Reward,GripResponse = [],[],[]
Subject = []
slope_data, age, sex, bisbas_bis, mdi, bisbas_bas = [], [], [], [], [], []
accumbensL_roi, accumbensR_roi, putamenL_roi, putamenR_roi, caudateL_roi, caudateR_roi = [[] for x in range(6)]

from nilearn import datasets
bg = datasets.load_mni152_template(resolution=1)
mask_gm = datasets.load_mni152_gm_mask()

grey_matter_masker = NiftiMasker(mask_img=mask_gm)

for sub in beh_ref.id:
    for run_nr,run in enumerate(runs):
        for rew_nr,rew in enumerate(range(-1,2,2)):
            for hand_nr,hand in enumerate(range(-1,2,2)):
                
                Run.append(run_nr+1)
                Reward.append(rew)
                GripResponse.append(hand)
                Subject.append(sub)
                
                if rew_nr == 0:
                    if hand == -1:
                        img = load_img(os.path.join(image_dir,
                        f'sub-0{sub}/ses-PRISMA/{run}/first_level_maps/sub-0{sub}_contrast-golowleft_stat-effect_size_statmap.nii.gz'))
                    elif hand == 1:
                        img = load_img(os.path.join(image_dir,
                        f'sub-0{sub}/ses-PRISMA/{run}/first_level_maps/sub-0{sub}_contrast-golowright_stat-effect_size_statmap.nii.gz'))
                elif rew_nr == 1:
                    if hand == -1:
                        img = load_img(os.path.join(image_dir,
                        f'sub-0{sub}/ses-PRISMA/{run}/first_level_maps/sub-0{sub}_contrast-gohighleft_stat-effect_size_statmap.nii.gz'))
                    elif hand == 1:
                        img = load_img(os.path.join(image_dir,
                        f'sub-0{sub}/ses-PRISMA/{run}/first_level_maps/sub-0{sub}_contrast-gohighright_stat-effect_size_statmap.nii.gz'))
                
                img_gm_data = grey_matter_masker.fit_transform(img)
                img_gm = grey_matter_masker.inverse_transform(img_gm_data)
                
                accumbensL_data = masker_accumbens_L_mask.fit_transform(img_gm)
                accumbensL_roi.append(np.mean(accumbensL_data))
                
                accumbensR_data = masker_accumbens_R_mask.fit_transform(img_gm)
                accumbensR_roi.append(np.mean(accumbensR_data))
                
                putamenL_data = masker_putamen_L_mask.fit_transform(img_gm)
                putamenL_roi.append(np.mean(putamenL_data))
                
                putamenR_data = masker_putamen_R_mask.fit_transform(img_gm)
                putamenR_roi.append(np.mean(putamenR_data))
                
                caudateL_data = masker_caudate_L_mask.fit_transform(img_gm)
                caudateL_roi.append(np.mean(caudateL_data))
                
                caudateR_data = masker_caudate_R_mask.fit_transform(img_gm)
                caudateR_roi.append(np.mean(caudateR_data))
                
                behav_code = beh_ref[beh_ref['id'] == sub]['Behav']
                
                sub_data = beh_file[(beh_file.id == behav_code.values[0]) & (beh_file.Correct == 1) & (beh_file.GripResponse == hand) &
                    (beh_file.Mirror == 0) & (beh_file.Run == run_nr+1) & (beh_file.RewardPromise == rew) & (beh_file.Slope > 0)]

                slope_mean = sub_data['Slope'].mean()
                age_mean = sub_data['age'].mean()
                sex_mean = sub_data['gender'].mean()
                bisbasbis_mean = sub_data['BISBAS_bis'].mean()
                bisbasbas_mean = sub_data['BISBAS_bas'].mean()
                mdi_mean = sub_data['MDI'].mean()
                
                slope_data.append(slope_mean)
                age.append(age_mean)
                sex.append(sex_mean)
                bisbas_bis.append(bisbasbis_mean)
                bisbas_bas.append(bisbasbas_mean)
                mdi.append(mdi_mean)
                
data = pd.DataFrame({
    'SubjectID': Subject,
    'Run': Run,
    'Reward': Reward,
    'GripResponse': GripResponse,
    'age': age,
    'sex': sex,
    'bisbas_bis': bisbas_bis,
    'bisbas_bas': bisbas_bas,
    'mdi': mdi,
    'Slope': slope_data,
    'AccumbensL': accumbensL_roi,
    'AccumbensR': accumbensR_roi,
    'PutamenL': putamenL_roi,
    'PutamenR': putamenR_roi,
    'CaudateL': caudateL_roi,
    'CaudateR': caudateR_roi
})

data.to_csv('roi_outputs_gm.csv',index=False)

# Start stats I do not understand
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

def run_mixed_effects_model(endog, exog, group):
    exog = sm.add_constant(exog)
    model = MixedLM(endog, exog, groups=group)
    result = model.fit()
    return result

import statsmodels.formula.api as smf

for i, (roi, title) in enumerate(zip(['AccumbensL', 'AccumbensR',
                                      'PutamenL', 'PutamenR',
                                      'CaudateL', 'CaudateR'], 
                                     ['Accumbens L', 'Accumbens R', 
                                      'Putamen L', 'Putamen R',
                                      'Caudate L', 'Caudate R'])):
    
    formula = f'{roi} ~ Slope * Reward + age + sex + Run'
    
    model = smf.mixedlm(formula, data, groups=data['SubjectID'], 
                        re_formula="~Run", vc_formula={"SubjectID:Run": "0 + Run"})
    
    result = model.fit()
    print(f'{title} Model Summary:')
    print(result.summary())


# for i, (roi, title) in enumerate(zip(['StriatumL', 'StriatumR',
#                                       'PutamenL', 'PutamenR',
#                                       'CaudateL','CaudateR'], 
                                     
#                                      ['Striatum L', 'Striatum R', 
#                                       'Putamen L','Putamen R',
#                                       'Caudate L','Caudate R'])):
    
#     exog = data[['Slope', 'age', 'sex','GripResponse','Run','Reward']]
#     endog = data[roi]
#     group = data['SubjectID']
#     model = run_mixed_effects_model(endog, exog, group)
#     print(model.summary())

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# for i, (roi, title) in enumerate(zip(['striatumL_roi', 'striatumR_roi', 'mPFC_roi'], ['Striatum L', 'Striatum R', 'mPFC'])):
#     exog = data[['behavioral_data', 'age', 'sex', 'bisbas']]
#     endog = data[roi]
#     group = data['subject']
#     model = run_mixed_effects_model(endog, exog, group)
    
#     sorted_indices = np.argsort(data['behavioral_data'])
#     sorted_behavioral_data = data['behavioral_data'].iloc[sorted_indices]
#     sorted_exog = exog.iloc[sorted_indices]
#     sorted_fitted_values = model.predict(sm.add_constant(sorted_exog))
    
#     axes[i].scatter(data['behavioral_data'], data[roi], color='b')
#     axes[i].plot(sorted_behavioral_data, sorted_fitted_values, 'r')
#     axes[i].set_title(f'Behavioral Slope vs {title}')
#     axes[i].set_xlabel('Behavioral Slope')
#     axes[i].set_ylabel(f'Mean {title}')
#     axes[i].legend()

# plt.tight_layout()
# plt.show()
