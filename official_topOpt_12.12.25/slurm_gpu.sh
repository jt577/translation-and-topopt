#!/bin/bash
#SBATCH -J top_opt -o slurm_out.o%j
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4090

# Activate virtual environment
source /data2/jt577/COSMO-RS/myenv/bin/activate

# Run python script: python /path/to/main.py /path/to/infile /path/to/outdir
python /data2/jt577/2025/December_2025/topopt/official_topOpt_12.12.25 infile.txt out