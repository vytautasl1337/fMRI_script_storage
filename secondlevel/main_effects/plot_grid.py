import matplotlib,os
import matplotlib.pyplot as plt

working_dir = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633'
picture_input = os.path.join(working_dir,'derivatives/task_output/SecondLevel/Main_effects/fpr_0.001')

images = ['go','leftgo','rightgo']
contrast=['Go','Left Go','Right Go']
runs = ['run_1','run_2','run_3','run_12','run_123']
run_id = ['Run 1','Run 2','Run 3','Run 1+2','Run 1+2+3']

#####################################################################
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison1.png', dpi=300)


#################################################
images = ['nogo','high','low']
contrast=['NoGo','High reward','Low reward']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison2.png', dpi=300)


#################################################
images = ['sad','happy','neutral']
contrast=['Sad faces','Happy faces','Neutral faces']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison3.png', dpi=300)

#################################################
images = ['win','loss','winloss']
contrast=['Win','Loss','Win>Loss']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison4.png', dpi=300)

#################################################
images = ['gonogo','leftright','rightleft']
contrast=['Go>NoGo','Left>Right','Right>Left']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison5.png', dpi=300)


#################################################
images = ['sadneutral','happyneutral','sadhappy']
contrast=['Sad faces>Neutral faces','Happy faces>Neutral faces','Sad faces>Happy faces']
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for row in range(3):
    for column in range(5):
        ax = axes[row, column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison6.png', dpi=300)


#################################################
images = ['highlow']
contrast=['High reward block >Low reward block']
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
for row in range(1):
    for column in range(5):
        ax = axes[column]
        
        if row == 0:
            ax.set_title(f"{run_id[column]}", fontweight='bold', fontsize=16)
        
        if column == 0:
            ax.text(0, 0, contrast[row], rotation=0, fontsize=15, ha='center', va='center')
            ax.axis('off')
        
        image = plt.imread(os.path.join(picture_input, runs[column], images[row], f"{images[row]}.png"))
        
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()
plt.savefig('pictures/comparison7.png', dpi=300)







