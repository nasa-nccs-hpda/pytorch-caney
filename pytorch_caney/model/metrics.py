import logging
from typing import List

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module metrics
#
# General functions to compute custom metrics.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------

EPSILON = 1e-15


# ------------------------------ Metric Functions -------------------------- #

def iou_val(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def acc_val(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def prec_val(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro'), \
        precision_score(y_true, y_pred, average=None)


def recall_val(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro'), \
        recall_score(y_true, y_pred, average=None)


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def binary_mean_iou(
            logits: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:

    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


# -------------------------------------------------------------------------------
# module metrics Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
