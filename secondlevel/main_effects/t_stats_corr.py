import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os,glob
from nilearn.image import load_img
import nilearn
from scipy.stats import pearsonr
from nilearn.masking import compute_background_mask



working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/task_output/SecondLevel/Main_effects/fpr_0.001'

map_pack1 = ['run_1','run_1','run_1','run_1']
map_pack2 = ['run_2','run_3','run_12','run_123']
map_xlabel = 'Second level RUN 1'
map_ylabel = ['Second level RUN 2','Second level RUN 3','Second level RUN 1+2','Second level RUN 1+2+3']

contrast = ['gonogo','winloss','sadneutral','highlow']
titles = ['Go>NoGo','Win>Loss','Sad>Neutral','High>Low']
        
        
for index, con in enumerate(contrast):

    fig, axes = plt.subplots(1, 4, figsize=(30, 10))
    
    for column in range(4):
        
        t_map1 = load_img(os.path.join(working_dir, '{}/{}/{}_stat-stat_statmap.nii.gz'.format(map_pack1[column], con, con)))
        t_map2 = load_img(os.path.join(working_dir, '{}/{}/{}_stat-stat_statmap.nii.gz'.format(map_pack2[column], con, con)))
        
        # my_mask = NiftiMasker(mask)
        # my_mask.fit(t_map1)
        # x = my_mask.transform(t_map1)
        # my_mask.fit(t_map2)
        # y = my_mask.transform(t_map2)
        x=t_map1.get_data()
        y=t_map2.get_data()
        
        ax = axes[column]
        ax.set_title(f"{titles[index]}", fontweight='bold', fontsize=16)
        ax.set_xlabel(f"{map_xlabel}", fontweight='bold', fontsize=16)
        ax.set_ylabel(f"{map_ylabel[column]}", fontweight='bold', fontsize=16)
        
        xedges, yedges = np.linspace(-4, 4, 42), np.linspace(-25, 25, 42)
        hist, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=(xedges, yedges))

        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1] - 1)
        c = hist[xidx, yidx]

        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)

        c = hist[xidx, yidx]
        scatter = ax.scatter(x, y, c=c, cmap='viridis', alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frequency')
        
        # Calculate and plot the correlation line
        corr_coef, _ = pearsonr(x.flatten(), y.flatten())
        ax.plot(np.unique(x.flatten()), np.poly1d(np.polyfit(x.flatten(), y.flatten(), 1))(np.unique(x.flatten())), color='red')

        # Add the correlation coefficient value as text
        ax.text(np.min(x), np.max(y), f'Correlation coefficient: {corr_coef:.2f}', fontsize=12, color='red')

    plt.tight_layout()
    #plt.show()
    plt.savefig('pictures/corr_{}.png'.format(index+1))
    
    
map_pack1 = ['run_1','run_1']
map_pack2 = ['run_2','run_3']
map_xlabel = ['Average of T-statistics (Run 1+2)','Average of T-statistics (Run 1+3)']
map_ylabel = ['Difference of T-statistics (Run 1-2)','Difference of T-statistics (Run 1-3)']

contrast = ['gonogo','winloss','sadneutral','highlow']
titles = ['Go>NoGo','Win>Loss','Sad>Neutral','High>Low']

for index, con in enumerate(contrast):

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for column in range(2):
        
        t_map1 = load_img(os.path.join(working_dir, '{}/{}/{}_stat-stat_statmap.nii.gz'.format(map_pack1[column], con, con)))
        t_map2 = load_img(os.path.join(working_dir, '{}/{}/{}_stat-stat_statmap.nii.gz'.format(map_pack2[column], con, con)))
        
        # my_mask = NiftiMasker(mask)
        # my_mask.fit(t_map1)
        # x = my_mask.transform(t_map1)
        # my_mask.fit(t_map2)
        # y = my_mask.transform(t_map2)
        x_get=t_map1.get_data()
        y_get=t_map2.get_data()
        x = (x_get + y_get) / 2
        #y = np.abs(x_get - y_get)
        y = x_get - y_get
        
        ax = axes[column]
        ax.set_title(f"{titles[index]}", fontweight='bold', fontsize=16)
        ax.set_xlabel(f"{map_xlabel[column]}", fontweight='bold', fontsize=16)
        ax.set_ylabel(f"{map_ylabel[column]}", fontweight='bold', fontsize=16)
        
        xedges, yedges = np.linspace(-4, 4, 42), np.linspace(-25, 25, 42)
        hist, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=(xedges, yedges))

        xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1] - 1)
        c = hist[xidx, yidx]

        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)

        c = hist[xidx, yidx]
        scatter = ax.scatter(x, y, c=c, cmap='viridis', alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frequency')
        ax.axhline(y=np.mean(y), color='red', linestyle='--', label='Mean difference')
        # Calculate and plot the correlation line
        #corr_coef, _ = pearsonr(x.flatten(), y.flatten())
        #ax.plot(np.unique(x.flatten()), np.poly1d(np.polyfit(x.flatten(), y.flatten(), 1))(np.unique(x.flatten())), color='red')

        # Add the correlation coefficient value as text
        #ax.text(np.min(x), np.max(y), f'Correlation coefficient: {corr_coef:.2f}', fontsize=12, color='red')

    plt.tight_layout()
    #plt.show()
    plt.savefig('pictures/corr_mean_{}.png'.format(index+1))

# # Step 1: Calculate the mean T value
# mean_t_value = (x + y) / 2

# # Step 2: Calculate the absolute difference
# difference = np.abs(x - y)

# Step 3: Plot Bland-Altman Graph
# plt.figure(figsize=(8, 6))
# plt.scatter(mean_t_value.flatten(), difference.flatten(), color='blue', alpha=0.5)
# plt.axhline(y=np.mean(difference), color='red', linestyle='--', label='Mean Difference')
# plt.xlabel('Mean T Value')
# plt.ylabel('Absolute Difference')
# plt.title('Bland-Altman Plot')
# plt.legend()
# plt.grid(True)
# plt.show()



# plt.figure(figsize=(8, 6))
# plt.scatter(x.flatten(), y.flatten(), color='blue', alpha=0.5)
# plt.xlabel('t1')
# plt.ylabel('t2')
# plt.title('Bland-Altman Plot')
# plt.legend()
# plt.grid(True)
# plt.show()































fig, ax = plt.subplots()
ax.hist2d(mean_t_value.flatten(), difference.flatten())
plt.show()


from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 'blue'), (0.5, 'red'), (1, 'blue')]  # Blue to Red gradient
cmap_name = 'custom_map'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(difference.mean(axis=0), cmap=cm, interpolation='nearest', aspect='auto', extent=[mean_t_value.min(), mean_t_value.max(), difference.min(), difference.max()])
plt.colorbar(label='Absolute Difference')
plt.title('Bland-Altman Heatmap')
plt.xlabel('Mean T Value')
plt.ylabel('Absolute Difference')
plt.show()