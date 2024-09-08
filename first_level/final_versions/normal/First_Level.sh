#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=6G
#SBATCH -o /mnt/scratch/projects/PBCTI/errors/sbatch_reports/First_level/output%a.out
#SBATCH -e /mnt/scratch/projects/PBCTI/errors/sbatch_reports/First_level/error%a.err
echo "Hostname: $(hostname)"


PYTHON_BIN=/mrhome/vytautasl/anaconda3/envs/mri/bin/python
BIDS_DIRECTORY=/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/
SLURMID=$SLURM_ARRAY_TASK_ID
SUBID=$(printf "sub-0%d" $SLURMID)


(XDG_RUNTIME_DIR= /$PYTHON_BIN sbatch_first_level.py -s "${SUBID}" -bdir $BIDS_DIRECTORY )

echo "First Level analysis finished successfully"
