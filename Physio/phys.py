import os
import pandas as pd
import matplotlib.pyplot as plt
import CMRR_dicom_physread_v06b as physread

bidsbase_directory = '/mnt/scratch/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/'  #parent folder, for me contains source,raw and derivatives folders (raw folder has phys data inside beh subfolders.)

retroicor_directory = os.path.join(bidsbase_directory, "derivatives", "phys") # output folder
df_participants = pd.read_csv(os.path.join(bidsbase_directory, "participants.tsv"), sep="\t") #participants.tsv file created by fmriprep


for subid in df_participants.participant_id:

    for runid in ["rest_run-1",'ROSETTA_run-1','ROSETTA_run-2','ROSETTA_run-3']: #one rest run and 3 task runs
        
        file = os.path.join(
            bidsbase_directory,
            "raw/{subid}/ses-PRISMA/beh/{subid}_ses-PRISMA_task-{runid}_part-mag_phys.dcm".format(  #find the phys file (for subid, for each run in runid)
                subid=subid, runid=runid 
            ),
        )
        print(file)
        filename_output_basename = "{subid}_ses-PRISMA_task-{runid}_phys".format(subid=subid, runid=runid) #output file name
        output_directory = os.path.join(retroicor_directory, subid)  #make output folder subject specific

        assert os.path.exists(file), "File does not exist"

        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        physread.parseDCM(  #run magical Kristoffer script to take care of things
            file,
            outdir=output_directory,
            outname=filename_output_basename,
            resporder=1,
            pulseorder=2,
            fs=400.0,
            makeplot=True,
            mindistr=2.0,
            mindistp=0.5,
            respprom=40,
            pulseprom=20,
            prefix="f_",
            pulsepass=2.0,
            pulsestop=10.0,
            resppass=1.0,
            respstop=3.0,
        )
        plt.close("all")