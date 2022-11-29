#!/bin/bash
#SBATCH --job-name=synthetic_dim_16_nClusters_2_setting_1
#SBATCH --mem=16Gb
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16 # 16 threads per task
#SBATCH --partition=short
#SBATCH --output="output/synthetic_dim_16_nClusters_2_setting_1.%A.%a.out"
#SBATCH --error="output/synthetic_dim_16_nClusters_2_setting_1.%A.%a.err"
#SBATCH --array=0-24
eval "$(conda shell.bash hook)"
conda activate st2
cd /home/zeiberg.d/LEVERAGING_STRUCTURE_CODE_APPENDIX/
srun python runExperiment.py --experimentPath /scratch/zeiberg.d/leveragingStructureFinalExperiments/experiments/synthetic_dim_16_nClusters_2_setting_1_$SLURM_ARRAY_TASK_ID --datasetType synthetic --problem_name synthetic --problem_setting 1 --syntheticDim 16 --syntheticNClusters 2