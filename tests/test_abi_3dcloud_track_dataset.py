from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_caney.configs.config import _C
from pytorch_caney.datasets.abi_3dcloud_track_dataset import AbiToa3DCloudTrackDataset
from pytorch_caney.three_d_cloud_track_utils import (
    build_profile_valid_mask,
    channel_numbers_to_indices,
    rasterize_track_to_dense,
)


def make_config(tmp_path):
    config = _C.clone()
    config.defrost()
    config.DATA.IMG_SIZE = 8
    config.DATA.DATA_PATHS = [str(tmp_path)]
    config.DATA.TEST_DATA_PATHS = [str(tmp_path)]
    config.DATA.INPUT_CHANNELS = [2, 3, 1, 5, 6, 7, 4, 9, 10, 11, 12, 14, 15, 16]
    config.DATA.SOLAR_ZENITH_CHANNEL = 22
    config.DATA.CURTAIN_KEY = 'Cloud_mask'
    config.DATA.CURTAIN_VALID_KEY = 'cloud_mask_cpr_atlid_msi_1km'
    config.DATA.CURTAIN_HEIGHT_KEY = 'height'
    config.DATA.TRACK_ROW_KEY = 'chip_row_idx'
    config.DATA.TRACK_COL_KEY = 'chip_col_idx'
    config.DATA.CURTAIN_FILL_VALUE = None
    config.DATA.CURTAIN_EXCLUDE_TOP_BINS = 0
    config.DATA.CURTAIN_EXCLUDE_BOTTOM_BINS = 0
    config.DATA.CURTAIN_AGGREGATION = 'majority'
    config.DATA.CURTAIN_AGGREGATION_THRESHOLD = 0.5
    config.DATA.RETURN_METADATA = True
    config.MODEL.IN_CHANS = 14
    config.MODEL.OUTPUT_BINS = 4
    config.freeze()
    return config


def write_sample(path: Path, chip, curtain, valid_source, rows, cols, heights=None):
    if heights is None:
        heights = np.tile(np.arange(curtain.shape[1], dtype=np.float32), (curtain.shape[0], 1))
    payload = {
        'Cloud_mask': curtain,
        'cloud_mask_cpr_atlid_msi_1km': valid_source,
        'chip_row_idx': rows,
        'chip_col_idx': cols,
        'height': heights,
    }
    np.savez(path, chip=chip, data=np.array(payload, dtype=object))


def test_rasterize_diagonal_track_majority():
    curtain = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(curtain, dtype=bool)
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    label, dense_valid, *_ = rasterize_track_to_dense(curtain, rows, cols, valid, (4, 4))
    assert label[:, 0, 0].tolist() == [1, 0, 1, 0]
    assert label[:, 1, 1].tolist() == [0, 1, 1, 0]
    assert label[:, 2, 2].tolist() == [1, 1, 0, 0]
    assert dense_valid[:, 0, 0].all()


def test_rasterize_short_track_clips_corner():
    curtain = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
    valid = np.ones_like(curtain, dtype=bool)
    rows = np.array([-1, 7])
    cols = np.array([0, 7])
    label, dense_valid, kept_rows, kept_cols, track_mask = rasterize_track_to_dense(
        curtain,
        rows,
        cols,
        valid,
        (8, 8),
    )
    assert kept_rows.tolist() == [7]
    assert kept_cols.tolist() == [7]
    assert track_mask.sum() == 1
    assert label[:, 7, 7].tolist() == [0, 1, 0, 1]
    assert dense_valid[:, 7, 7].all()


def test_rasterize_horizontal_track():
    curtain = np.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
        dtype=np.float32,
    )
    valid = np.ones_like(curtain, dtype=bool)
    rows = np.array([2, 2, 2])
    cols = np.array([1, 2, 3])
    label, _, _, _, track_mask = rasterize_track_to_dense(curtain, rows, cols, valid, (8, 8))
    assert track_mask[2, 1]
    assert track_mask[2, 2]
    assert track_mask[2, 3]
    assert label[:, 2, 2].tolist() == [1, 1, 0, 0]


def test_rasterize_empty_track():
    curtain = np.zeros((0, 4), dtype=np.float32)
    valid = np.zeros((0, 4), dtype=bool)
    rows = np.zeros((0,), dtype=np.int64)
    cols = np.zeros((0,), dtype=np.int64)
    label, dense_valid, kept_rows, kept_cols, track_mask = rasterize_track_to_dense(
        curtain,
        rows,
        cols,
        valid,
        (8, 8),
    )
    assert label.shape == (4, 8, 8)
    assert not dense_valid.any()
    assert kept_rows.size == 0
    assert kept_cols.size == 0
    assert not track_mask.any()


def test_duplicate_profiles_majority_any_and_threshold():
    curtain = np.array(
        [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    valid = np.ones_like(curtain, dtype=bool)
    rows = np.array([3, 3, 3])
    cols = np.array([4, 4, 4])

    majority_label, majority_valid, *_ = rasterize_track_to_dense(
        curtain,
        rows,
        cols,
        valid,
        (8, 8),
        aggregation='majority',
    )
    assert majority_label[:, 3, 4].tolist() == [1, 1, 0, 0]
    assert majority_valid[:, 3, 4].all()

    any_label, _, *_ = rasterize_track_to_dense(
        curtain,
        rows,
        cols,
        valid,
        (8, 8),
        aggregation='any',
    )
    assert any_label[:, 3, 4].tolist() == [1, 1, 0, 1]

    threshold_label, _, *_ = rasterize_track_to_dense(
        curtain,
        rows,
        cols,
        valid,
        (8, 8),
        aggregation='threshold',
        threshold=0.75,
    )
    assert threshold_label[:, 3, 4].tolist() == [0, 0, 0, 0]


def test_fill_values_and_invalid_bins():
    curtain = np.array([[1, -1, 0, 1], [0, -1, 1, 0]], dtype=np.float32)
    valid_source = np.array([[1.0, np.nan, 1.0, 1.0], [1.0, np.nan, 1.0, 1.0]], dtype=np.float32)
    valid = build_profile_valid_mask(
        curtain,
        validity_source=valid_source,
        fill_value=-1,
        exclude_top_bins=1,
        exclude_bottom_bins=1,
    )
    assert valid.tolist() == [[False, False, True, False], [False, False, True, False]]


def test_dataset_collates_variable_track_lengths(tmp_path):
    config = make_config(tmp_path)
    raw_channels = 22
    chip = np.zeros((8, 8, raw_channels), dtype=np.float32)
    chip[..., 21] = 45.0

    curtain_a = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    valid_a = np.ones_like(curtain_a, dtype=np.float32)
    rows_a = np.array([0, 1], dtype=np.int32)
    cols_a = np.array([0, 1], dtype=np.int32)
    write_sample(tmp_path / 'sample_a.npz', chip, curtain_a, valid_a, rows_a, cols_a)

    curtain_b = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]], dtype=bool)
    valid_b = np.ones_like(curtain_b, dtype=np.float32)
    rows_b = np.array([2, 3, 4], dtype=np.int32)
    cols_b = np.array([2, 3, 4], dtype=np.int32)
    write_sample(tmp_path / 'sample_b.npz', chip, curtain_b, valid_b, rows_b, cols_b)

    dataset = AbiToa3DCloudTrackDataset(config, config.DATA.DATA_PATHS, transform=None)
    loader = DataLoader(dataset, batch_size=2)
    images, labels, valid, metadata = next(iter(loader))

    assert images.shape == (2, 8, 8, 14)
    assert labels.shape == (2, 4, 8, 8)
    assert valid.shape == (2, 4, 8, 8)
    assert metadata['track_rows'].shape == (2, 64)
    assert metadata['track_length'].tolist() == [2, 3]


def test_channel_selection_uses_requested_order():
    indices = channel_numbers_to_indices([2, 3, 1, 5])
    assert indices == [1, 2, 0, 4]
