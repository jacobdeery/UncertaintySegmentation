#!/usr/bin/env bash

CONFIG_DIR="$(dirname $0)/mask_rcnn"
CONFIG_FILE="$CONFIG_DIR/mask_rcnn_cityscapes.yaml"

python mask_rcnn/train_net.py --config-file $CONFIG_FILE --eval-only
