import numpy as np
from torchvision.ops import nms


def get_uncertainty_exist(instances):
    keep = nms(instances.pred_boxes, instances.scores, 0.5)

    masks_orig = np.asarray(instances.pred_masks).astype(int)
    num_inst_orig = np.sum(masks_orig, axis=0)
    num_inst_nms = np.sum(masks_orig[keep], axis=0)

    return np.nan_to_num(num_inst_orig - num_inst_nms)


def get_uncertainty_centroid(masks):
    h, w = masks[0].shape

    total_masks = len(masks)

    centroids = np.full((h, w, total_masks, 2), np.nan)
    num_inst = np.zeros((h, w))

    x_unc = np.zeros((h, w, 1))
    y_unc = np.zeros((h, w, 1))

    x_coords = np.arange(w)
    y_coords = np.arange(h)
    xx, yy = np.meshgrid(x_coords, y_coords)

    for i, mask in enumerate(masks):
        array_mask = np.logical_not(mask)
        x_cent = np.mean(np.ma.masked_array(xx, mask=array_mask))
        y_cent = np.mean(np.ma.masked_array(yy, mask=array_mask))
        centroids[:, :, i, :][mask == 1] = (x_cent, y_cent)
        num_inst[mask == 1] += 1

    stds = np.nanstd(centroids, axis=2)
    x_unc = stds[:, :, 0]
    y_unc = stds[:, :, 1]

    # return np.maximum(num_inst - 1, np.zeros(num_inst.shape))
    return np.nan_to_num(x_unc + y_unc)
