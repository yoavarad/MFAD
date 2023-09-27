#!/bin/csh
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH -c 10
#SBATCH --mem=20g
#SBATCH --time=20-0
#SBATCH --mail-user=yoavarad10@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#aaaaaSBATCH --mail-type=ALL
#SBATCH --output='/cs/labs/werman/yoavarad/VideoMAEv2/slurm_scripts/logs/%x-%j'
#SBATCH --job-name=extract_tad_feature-$DATASET-$MODE-$HEAD

cd /cs/labs/werman/yoavarad/VideoMAEv2
source venv-VMAE/bin/activate.csh

module load cuda/11.3
module load gcc/9.3.0

python extract_tad_feature.py --data_set '$DATASET' \
                              --data_path 'data/$DATASET/$MODEing/videos' \
                              --save_path 'extracted_features/continuous/$DATASET/$MODEing' \
                              --model 'vit_giant_patch14_224' \
                              --ckpt_path 'model_weights/vit_g_hybrid_pt_1200e_$HEAD_ft.pth'
