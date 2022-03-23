#!/usr/bin/env bash

CONFIG_DIR="$(dirname $0)/mask_rcnn"
CONFIG_FILE="$CONFIG_DIR/mask_rcnn_cityscapes.yaml"

python3 mask_rcnn/mrcnn_calc_uncertainty.py --config-file $CONFIG_FILE
