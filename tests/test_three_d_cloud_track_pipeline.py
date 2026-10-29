import torch

from pytorch_caney.configs.config import _C
from pytorch_caney.pipelines.three_d_cloud_track_pipeline import ThreeDCloudTrackTask


def make_pipeline_config():
    config = _C.clone()
    config.defrost()
    config.DATA.IMG_SIZE = 128
    config.MODEL.IN_CHANS = 14
    config.MODEL.ENCODER = 'fcn'
    config.MODEL.DECODER = 'denseprofiledecoder'
    config.MODEL.HEAD = 'dense_profile_head'
    config.MODEL.OUTPUT_BINS = 206
    config.MODEL.HEAD_KERNEL_SIZE = 1
    config.MODEL.HEAD_DROPOUT = 0.0
    config.LOSS.NAME = 'bce'
    config.freeze()
    return config


def test_forward_shape_matches_dense_profile_output():
    config = make_pipeline_config()
    model = ThreeDCloudTrackTask(config)
    x = torch.randn(2, 14, 128, 128)
    y = model(x)
    assert y.shape == (2, 206, 128, 128)
