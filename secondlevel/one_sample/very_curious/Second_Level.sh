#!/bin/bash
#SBATCH --partition=HPC
#SBATCH -o /mnt/scratch/projects/PBCTI/errors/sbatch_reports/Second_level/output.%a.out # STDOUT
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vytautasl@drcmr.dk
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G


PYTHON_BIN=/mrhome/vytautasl/anaconda3/envs/mri/bin/python
BIDS_DIRECTORY=/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/
SLURMID=$SLURM_ARRAY_TASK_ID


# Run second level
(XDG_RUNTIME_DIR= /$PYTHON_BIN SecondLevel_onesample.py -bdir $BIDS_DIRECTORY -ji $SLURM_ARRAY_TASK_ID)


