#!/bin/bash
#SBATCH --job-name=amazon_reviews_multi_setting_2
#SBATCH --mem=16Gb
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16 # 16 threads per task
#SBATCH --partition=short
#SBATCH --output="output/amazon_reviews_multi_setting_2.%A.%a.out"
#SBATCH --error="output/amazon_reviews_multi_setting_2.%A.%a.err"
#SBATCH --array=0-24
eval "$(conda shell.bash hook)"
conda activate st2
cd /home/zeiberg.d/LEVERAGING_STRUCTURE_CODE_APPENDIX/
srun python runExperiment.py --experimentPath /scratch/zeiberg.d/leveragingStructureFinalExperiments/experiments/amazon_reviews_setting_2_$SLURM_ARRAY_TASK_ID --datasetType huggingface --problem_name amazon_reviews_multi --problem_setting 2 --pca False --cache_dir /scratch/zeiberg.d/leveragingStructureFinalExperiments/huggingface/ --huggingfaceSaveDir /scratch/zeiberg.d/leveragingStructureFinalExperiments/