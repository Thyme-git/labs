import numpy as np
import torch


def compute_iou(bbox1, bbox2):

    # TODO Compute IoU of 2 bboxes.

    x1 = torch.max(bbox1[:, 0], bbox2[:, 0])
    y1 = torch.max(bbox1[:, 1], bbox2[:, 1])
    x2 = torch.min(bbox1[:, 2], bbox2[:, 2])
    y2 = torch.min(bbox1[:, 3], bbox2[:, 3])

    interArea = torch.sum(torch.max(torch.zeros_like(x1), x2-x1+1)*torch.max(torch.zeros_like(y1), y2-y1+1))
    area1 = torch.sum(bbox1[:, 2]-bbox1[:, 0]+1)*(bbox1[:, 3]-bbox1[:, 1]+1)
    area2 = torch.sum(bbox2[:, 2]-bbox2[:, 0]+1)*(bbox2[:, 3]-bbox2[:, 1]+1)

    return interArea/(area1+area2-interArea)
    # End of todo
