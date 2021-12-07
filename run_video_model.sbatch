#!/bin/bash
  
#SBATCH --cpus-per-task=20
#SBATCH --time=36:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:v100:1

module purge
singularity exec --nv --overlay \
/scratch/trn219/capstone/trn219_torch.ext3:ro \
            /scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif \
            bash -c "
source /ext3/env.sh
python -u video_modeling.py --device cuda --output_string adam_lr001_optflownorm --batch_size 4 --loss WeightedMSE --lr 0.001 --momentum 0.0 --optical_flow --back_sub
exit
"

