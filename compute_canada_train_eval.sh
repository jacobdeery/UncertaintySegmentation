#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=MaskRCNN-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:t4:2                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --mail-type=ALL
#SBATCH --array=1   # 4 is the number of jobs in the chain

module load singularity/3.8

SING_IMG=detectron2.sif

PROJ_DIR=$PWD
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/cityscapes

mkdir $SLURM_TMPDIR/data
TMP_DATA_DIR=$SLURM_TMPDIR/data

tar -zxf $DATA_DIR/cityscapes.tar.gz -C $TMP_DATA_DIR

BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
singularity exec
--nv
--env-file $PROJ_DIR/envfile
--bind $PROJ_DIR:/MRCNN/code
--bind $TMP_DATA_DIR:/MRCNN/datasets
$SING_IMG
"

CODE_DIR=/MRCNN/code/mask_rcnn

TRAIN_CMD="$BASE_CMD
python $CODE_DIR/train_net.py
--config-file $CODE_DIR/modeling/mask_rcnn_cityscapes.yaml
--num-gpus 2
"

# TEST_CMD="$BASE_CMD bash"
# eval $TEST_CMD

eval $TRAIN_CMD
