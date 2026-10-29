# EarthCARE Dense-Track 3D Cloud Task

This document explains the new EarthCARE-specific 3D cloud retrieval path added to `pytorch-caney`.

For a short mixed conceptual/technical walkthrough in PDF form, see:

`docs/earthcare_dense_track_explainer.pdf`

## Problem

The existing 3D cloud task in `pytorch-caney` was designed around fixed-size CloudSat/CALIPSO curtains.

That does not work for the EarthCARE data in `/p/lustre5/smalley5/ecg_output` because:

- each sample contains a `128 x 128` GOES ABI chip
- each paired EarthCARE curtain has variable along-track length `L`
- default PyTorch collation fails when trying to stack `(L, Z)` curtains of different `L`
- the existing segmentation head upsamples to a fixed `(91, 40)` target that is not tied to the track geometry

## New Approach

Instead of predicting one resized curtain per sample, the new path predicts a vertical profile at every ABI pixel.

- model output: `(B, Z, 128, 128)`
- label volume: `(B, Z, 128, 128)`
- validity mask: `(B, Z, 128, 128)`

The dataset rasterizes each variable-length EarthCARE curtain into dense fixed-shape tensors. Loss and metrics are computed only where the validity mask is `True`.

This keeps the EarthCARE along-track dimension variable in the source data without padding or resampling the curtain itself.

## Data Assumptions

The new implementation was built against sample files in `/p/lustre5/smalley5/ecg_output/*.npz`.

Observed sample structure:

- image key: `chip`
- image layout: HWC
- image sample shape: `(128, 128, 22)`
- solar zenith angle: raw channel 22
- curtain key: `data["Cloud_mask"]`
- curtain shape: `(L, Z)`
- height bins: `Z = 206`
- track row indices: `data["chip_row_idx"]`
- track column indices: `data["chip_col_idx"]`
- validity source: `~isnan(data["cloud_mask_cpr_atlid_msi_1km"])`
- height grid: `data["height"]`, observed from `20000 m` to `-500 m` in `100 m` increments

## Default Channel Selection

The new EarthCARE path uses an explicit 14-channel ABI subset by default.

Requested ABI channels in order:

```text
[2, 3, 1, 5, 6, 7, 4, 9, 10, 11, 12, 14, 15, 16]
```

These are configured in `config.DATA.INPUT_CHANNELS` and are selected from the raw 22-channel chip before `AbiToaTransform` runs.

Channel 22 is not used as model input. It is only used for day/night plotting via solar zenith angle.

## New Components

### Dataset

File:

`pytorch_caney/datasets/abi_3dcloud_track_dataset.py`

Adds `AbiToa3DCloudTrackDataset`.

Responsibilities:

- loads `.npz` chips and EarthCARE curtains
- asserts the chip already matches `IMG_SIZE` so image/label alignment stays valid
- selects the configured 14 ABI channels
- builds dense `(Z, H, W)` labels and validity masks
- supports duplicate track hits within one ABI pixel using configurable aggregation
- exposes a no-op hook for future pixel offsets / parallax correction
- optionally returns collate-safe metadata

### Datamodule

File:

`pytorch_caney/datamodules/abi_3dcloud_track_datamodule.py`

Adds `AbiToa3DCloudTrackDataModule`.

Registered name:

`abitoa3dcloud_track`

The new dataset returns fixed-shape tensors, so the default DataLoader collate works without special handling.

### Decoder

File:

`pytorch_caney/models/decoders/dense_profile_decoder.py`

Adds `DenseProfileDecoder`.

Registered name:

`denseprofiledecoder`

Why this was needed:

- the existing `FcnDecoder` expands `128 x 128` inputs to `256 x 256`
- the dense EarthCARE supervision must stay at chip resolution `128 x 128`

This decoder upsamples encoded features directly to the configured chip size and then refines them with convolutions.

### Head

File:

`pytorch_caney/models/heads/dense_profile_head.py`

Adds `DenseProfileHead`.

Registered name:

`dense_profile_head`

Behavior:

- optional dropout
- `Conv2d(decoder_channels, Z, kernel_size)`
- no `Upsample`
- asserts that output spatial size matches the chip size

### Pipeline

File:

`pytorch_caney/pipelines/three_d_cloud_track_pipeline.py`

Adds `ThreeDCloudTrackTask`.

Registered name:

`3dcloud_track`

Behavior:

- output shape: `(B, Z, 128, 128)`
- masked BCE-with-logits loss with `reduction="none"`
- loss is computed in `float32`, even under autocast
- valid-only metrics:
  - IoU
  - precision
  - recall
  - F1
- metrics are accumulated over the epoch and logged at epoch end
- optional per-height IoU logging is supported via config

### Utilities

File:

`pytorch_caney/three_d_cloud_track_utils.py`

Contains:

- channel-number to zero-based-index conversion
- validity-mask construction
- per-pixel track rasterization
- duplicate-profile aggregation
- masked BCE helper
- masked metric tensor helper
- dense-to-curtain extraction
- per-height IoU accumulation helper

### Plotting

Files:

- `pytorch_caney/plotting/three_d_cloud_track.py`
- `python_examples/plot_3dcloud_track_predictions.py`

Behavior:

- daytime: true color
- nighttime: ABI channel 14
- day/night rule: mean solar zenith angle from raw channel 22
- overlay the EarthCARE track on the chip
- compare truth curtain, predicted curtain, and difference

## Config Defaults Added

New defaults were added in `pytorch_caney/configs/config.py` for:

- input ABI channel selection
- solar zenith channel
- curtain keys
- valid/fill handling
- top/bottom excluded bins
- duplicate-profile aggregation rule and threshold
- dense-head configuration
- output bin count
- masked loss positive weighting
- optional per-height IoU logging
- plotting day/night and display channels

## New Example Config

File:

`configs/3dcloudtask_fcn_dense_track_test.yaml`

This config uses:

- pipeline: `3dcloud_track`
- datamodule: `abitoa3dcloud_track`
- encoder: `fcn`
- decoder: `denseprofiledecoder`
- head: `dense_profile_head`
- output bins: `206`
- your requested 14-channel ABI subset

## Train Command

```bash
python pytorch_caney/ptc_cli.py --config-path configs/3dcloudtask_fcn_dense_track_test.yaml
```

## Plotting Command

```bash
python python_examples/plot_3dcloud_track_predictions.py \
  --config-path configs/3dcloudtask_fcn_dense_track_test.yaml \
  --checkpoint-path <checkpoint.ckpt> \
  --output-path outputs/earthcare_dense_track_plots.pdf \
  --num-samples 8
```

## Tests Added

New pytest coverage was added under `tests/`.

Covered cases:

- diagonal track rasterization
- corner clipping
- horizontal track
- empty track
- duplicate profiles in one pixel
- fill / invalid bins
- default DataLoader collation across different track lengths
- masked loss invariance at invalid positions
- zero gradients at invalid positions
- metric masking behavior
- dense forward output shape

Run the tests with:

```bash
pytest
```

Current result after implementation:

```text
12 passed
```

## Notes and Tradeoffs

- The existing CloudSat/CALIPSO 3D-cloud code path was intentionally left unchanged.
- The dense EarthCARE path needed a new decoder because the current FCN decoder outputs `256 x 256`, not `128 x 128`.
- The dataset asserts that chips already match `IMG_SIZE`. This avoids silent image/label misalignment caused by resizing only the image.
- Any future spatial augmentation must be applied identically to image, label, and validity mask.
- A hook exists for future per-profile pixel offsets, but no correction is applied yet.

## File Summary

### Added

- `configs/3dcloudtask_fcn_dense_track_test.yaml`
- `python_examples/plot_3dcloud_track_predictions.py`
- `pytorch_caney/datamodules/abi_3dcloud_track_datamodule.py`
- `pytorch_caney/datasets/abi_3dcloud_track_dataset.py`
- `pytorch_caney/models/decoders/dense_profile_decoder.py`
- `pytorch_caney/models/heads/dense_profile_head.py`
- `pytorch_caney/pipelines/three_d_cloud_track_pipeline.py`
- `pytorch_caney/plotting/three_d_cloud_track.py`
- `pytorch_caney/three_d_cloud_track_utils.py`
- `pytest.ini`
- `tests/conftest.py`
- `tests/test_abi_3dcloud_track_dataset.py`
- `tests/test_masked_metrics.py`
- `tests/test_three_d_cloud_track_pipeline.py`

### Updated

- `pytorch_caney/configs/config.py`
- `pytorch_caney/datamodules/__init__.py`
- `pytorch_caney/models/__init__.py`
- `pytorch_caney/models/decoders/__init__.py`
- `pytorch_caney/models/heads/__init__.py`
- `pytorch_caney/pipelines/__init__.py`
- `pytorch_caney/plotting/__init__.py`
