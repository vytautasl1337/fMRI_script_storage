import argparse

parser = argparse.ArgumentParser(description='Second level fMRI analysis')
parser.add_argument(
    "--bidsdir",
    "-bdir",
    help="set output width",
    default="/mnt/scratch/projects/PBCTI/BIDS_FINAL/BIDS_dataset_20230531-1633/",
)

parser.add_argument("--job_index","-ji", type=int, default=1,help='Job index')

args = parser.parse_args()


if args.bidsdir:
    bidsbase_directory = args.bidsdir

index = args.job_index

# call functions

from pmods import pmod_func
from normal import no_pmods

session='ses-PRISMA'
p_value = 0.01
run='run_123'

pmod_func(index,session,p_value,run)

no_pmods(index,session,p_value,run)

# publish
