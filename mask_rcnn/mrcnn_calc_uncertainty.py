#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import imp
import logging
import os
from collections import OrderedDict

import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.structures import Instances

from cityscapes_custom_evaluator import CityscapesPixelwiseInstanceEvaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return CityscapesPixelwiseInstanceEvaluator(dataset_name)


def wrap_model(model):
    num_runs = 5

    def run_model(inputs):
        model.eval()

        wrapped_outputs = [None] * len(inputs)

        scores = [[] for i in range(len(inputs))]
        classes = [[] for i in range(len(inputs))]
        masks = [[] for i in range(len(inputs))]
        img_sizes = [None] * len(inputs)

        import pdb; pdb.set_trace()

        for _ in range(num_runs):
            outputs = model(inputs)

            for i, output in enumerate(outputs):
                output = output['instances'].to('cpu')
                scores[i].extend(output.scores)
                classes[i].extend(output.pred_classes)

                masks[i].append(np.asarray(output.pred_masks))

                img_sizes[i] = output.image_size

        for i in range(len(inputs)):
            concat_masks = np.concatenate(masks[i], axis=2)
            wrapped_outputs[i] = Instances(img_sizes[i], scores=scores[i], pred_classes=classes[i], pred_masks=concat_masks)

        return wrapped_outputs

    return run_model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, wrap_model(model))
    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
