import pandas as pd
import matplotlib,os,numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

base_folder = '/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/derivatives/fmriprep-22.0.2/'

fd_table = pd.DataFrame()
for sub_nr in range(44):
    subject = 'sub-0{}'.format(sub_nr+1)
    confound_dir = os.path.join(base_folder,subject,'ses-PRISMA/func')
    for run in range(3):
        confound_file = pd.read_csv(os.path.join(confound_dir,
                                                 '{}_ses-PRISMA_task-ROSETTA_dir-ap_run-{}_part-mag_desc-confounds_timeseries.tsv'.format(subject,run+1)),sep='\t')
        fd = confound_file['framewise_displacement'].copy()
        fd_table=pd.concat([fd_table,fd],axis=1)
        
num_runs = 3

# Define the number of subjects
num_subjects = int(len(fd_table.columns)/num_runs)

# Define colors for each run
colors = ['b', 'g', 'r']

# Iterate over the runs
for run in range(num_runs):
    # Create a new figure for each run
    plt.figure(figsize=(10, 5))
    
    # Create a list to store the columns for the current run
    run_columns = []
    
    for col_idx in range(run, len(fd_table.columns), num_runs):
        # Append the values from the DataFrame column to a list
        run_values = fd_table.iloc[:, col_idx].tolist()
        # Append the list of values to the list of lists
        run_columns.append(run_values)
    

    # Plot the columns for the current run
    for i, lst in enumerate(run_columns):
        x = [i] * len(lst)  # Create x-values (all the same for each list)
        plt.scatter(x, lst, label=f'List {i + 1}')
        mean_val = numpy.nanmean(lst)
        print(mean_val)
        plt.scatter(i, mean_val, color='black', marker='o', s=100)


    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.title(f'Run {run + 1}')
    plt.xlabel('Subjects')
    plt.ylabel('FD value')
    #plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
        

session1_mean = fd_table.iloc[:, 0::3].mean()
session2_mean = fd_table.iloc[:, 1::3].mean()
session3_mean = fd_table.iloc[:, 2::3].mean()

fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize as needed

subjects = range(1, len(fd_table.columns) // 3 + 1)

# Plot scatter plot for each session
ax.scatter(subjects, session1_mean, label='Session 1 Mean', color='blue')
ax.scatter([s + 0.1 for s in subjects], session2_mean, label='Session 2 Mean', color='red')
ax.scatter([s + 0.2 for s in subjects], session3_mean, label='Session 3 Mean', color='green')

# Plot scatter plot for each session
for i in range(0, len(fd_table.columns), 3):
    session1 = fd_table.iloc[:, i]
    session2 = fd_table.iloc[:, i+1]
    session3 = fd_table.iloc[:, i+2]
    ax.scatter([s - 0.1 for s in subjects], session1, color='blue', alpha=0.3)
    ax.scatter(subjects, session2, color='red', alpha=0.3)
    ax.scatter([s + 0.1 for s in subjects], session3, color='green', alpha=0.3)

ax.set_xlabel('Subjects')
ax.set_ylabel('Mean FD')
ax.set_title('Mean session FD')
ax.legend()

plt.show()