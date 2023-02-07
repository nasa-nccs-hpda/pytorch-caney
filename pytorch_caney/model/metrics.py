import logging
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


# -------------------------------------------------------------------------------
# module metrics Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
