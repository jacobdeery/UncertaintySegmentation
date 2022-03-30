# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image
import matplotlib.cm as mcm

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation.evaluator import DatasetEvaluator

from .uncertainty import get_uncertainty_centroid, get_uncertainty_exist

class CityscapesEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using cityscapes API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )


class CityscapesPixelwiseInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                num_instances = len(output)
                inst_img = np.zeros(output.pred_masks[0].numpy().shape[0:2])

                # NOTE(jacob): Temporarily commenting this to reduce eval time
                # with open(pred_txt, "w") as fout:
                #     for i in range(num_instances):
                #         pred_class = int(output.pred_classes[i])
                #         classes = self._metadata.thing_classes[pred_class]
                #         class_id = name2label[classes].id
                #         score = output.scores[i]
                #         mask = output.pred_masks[i].numpy().astype("uint8")
                #         inst_img[mask == 1] = i + 1
                #         png_filename = os.path.join(
                #             self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                #         )

                #         Image.fromarray((mask * 255).astype("uint8")).save(png_filename)
                #         fout.write(
                #             "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                #         )

                img_in = input['image'].numpy().transpose([1, 2, 0])[:, :, ::-1]
                viz = Visualizer(img_in)
                inst_img = viz.draw_instance_predictions(output)
                inst_img_fname = os.path.join("/home/jacob/temp_results", basename + "_inst.png")
                inst_img.save(inst_img_fname)

                cmap = mcm.get_cmap('viridis')
                unc_img = get_uncertainty_centroid(output.pred_masks)
                unc_img = cmap(unc_img / (np.max(unc_img)))
                unc_img_fname = os.path.join("/home/jacob/temp_results", basename + "_unc_centroid.png")
                Image.fromarray((unc_img * 255).astype("uint8")).save(unc_img_fname)

                unc_img_exist = get_uncertainty_exist(output)
                unc_img_exist = cmap(unc_img_exist / (np.max(unc_img_exist)))
                unc_img_exist_fname = os.path.join("/home/jacob/temp_results", basename + "_unc_exist.png")
                Image.fromarray((unc_img_exist * 255).astype("uint8")).save(unc_img_exist_fname)
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import mask_rcnn.evaluation.pixelwise_instance_evaluation as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        self._working_dir.cleanup()
        return ret
