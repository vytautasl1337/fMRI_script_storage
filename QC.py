import nilearn, os, re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf


folder = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/FirstLevel/spm_pipeline/'
pdf_pages = pdf.PdfPages('gohighgolow_qc_spm.pdf')
contrast_paths = []

# Function to recursively search for files matching the pattern
def search_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('_contrast-gohighgolow_stat-effect_statmap.nii.gz'):
                contrast_paths.append(os.path.join(root, file))

search_files(folder)

# Loop and plot contrast maps for QC
from nilearn.image import load_img
from nilearn.plotting import plot_stat_map
bg = nilearn.image.load_img = '/mnt/projects/PBCTI/ds-MRI-study/templates/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
for i in contrast_paths:
    match = re.search(r'sub-(\d+)', i)
    subject_id = match.group(1)
    image = load_img(i)
    plot_stat_map(image, title = subject_id, bg_img=bg)
    pdf_pages.savefig()
    plt.close()
    
pdf_pages.close()







    
    