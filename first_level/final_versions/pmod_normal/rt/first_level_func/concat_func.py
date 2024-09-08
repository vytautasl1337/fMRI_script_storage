from nilearn.image import concat_imgs
import os
import glob
from nilearn.image import load_img

# Concat will concatinate all found images.

def concat_func_runs(combinations,index,subject_output,nifti_name,
                     subject_id,session,space_label,task_label):
    
    analyze_runs = combinations[index]
    print('Analyzing run(s)', analyze_runs)

    if len(analyze_runs) == 1:
        
        run_id = analyze_runs[0]
        print('Loading singular run', run_id)
        fmri_task = [load_img(os.path.join(subject_output,nifti_name.format(subject_id,
                                                                        session,
                                                                        task_label,
                                                                        run_id,
                                                                        space_label)))]
    else:
        funcs_grabbed=[]
        print('Concatinating functional runs into a list...')
        
        for run_id in analyze_runs:

            funcs_grabbed.extend(glob.glob(os.path.join(subject_output,nifti_name.format(subject_id,
                                                                                        session,
                                                                                        task_label,
                                                                                        run_id,
                                                                                        space_label))))
        print(funcs_grabbed)
        funcs_grabbed.sort()
        fmri_task = [concat_imgs(x) for x in funcs_grabbed] 
    
    return fmri_task