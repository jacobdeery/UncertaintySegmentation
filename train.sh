#!/usr/bin/env bash

CONFIG_DIR="$(dirname $0)/mask_rcnn/modeling"
CONFIG_FILE="$CONFIG_DIR/mask_rcnn_cityscapes.yaml"

python3 mask_rcnn/train_net.py --config-file $CONFIG_FILE
