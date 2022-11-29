#!/bin/bash
#SBATCH --job-name=employment
#SBATCH --mem=16Gb
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16 # 16 threads per task
#SBATCH --partition=short
#SBATCH --output="output/employment.%A.%a.out"
#SBATCH --error="output/employment.%A.%a.err"
#SBATCH --array=0-25
eval "$(conda shell.bash hook)"
conda activate st2
cd /home/zeiberg.d/LEVERAGING_STRUCTURE_CODE_APPENDIX/

srun python runExperiment.py --root_dir /scratch/zeiberg.d/folktables/ --experimentPath /scratch/zeiberg.d/leveragingStructureResponseExperiments/experiments/employment_setting_2_$SLURM_ARRAY_TASK_ID --datasetType acs --problem_name employment