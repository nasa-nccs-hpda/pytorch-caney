from collections import defaultdict
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


IGNORE_INDEX = -1


def channel_numbers_to_indices(channel_numbers: Sequence[int]) -> list[int]:
    return [int(channel) - 1 for channel in channel_numbers]


def apply_track_pixel_offsets(rows, cols, payload, config):
    # Hook for future per-profile parallax or geolocation corrections.
    return rows, cols


def build_profile_valid_mask(
    curtain: np.ndarray,
    validity_source: Optional[np.ndarray] = None,
    fill_value=None,
    exclude_top_bins: int = 0,
    exclude_bottom_bins: int = 0,
) -> np.ndarray:
    valid = np.ones_like(curtain, dtype=bool)

    if np.issubdtype(curtain.dtype, np.floating):
        valid &= ~np.isnan(curtain)

    if fill_value is not None:
        valid &= curtain != fill_value

    if validity_source is not None:
        validity_source = np.asarray(validity_source)
        if validity_source.shape != curtain.shape:
            raise ValueError(
                'Curtain validity source shape must match curtain shape. '
                f'Got {validity_source.shape} and {curtain.shape}.'
            )
        if validity_source.dtype == np.bool_:
            valid &= validity_source
        else:
            valid &= ~np.isnan(validity_source)

    if exclude_top_bins:
        valid[:, :exclude_top_bins] = False

    if exclude_bottom_bins:
        valid[:, -exclude_bottom_bins:] = False

    return valid


def aggregate_profile_bins(
    labels: np.ndarray,
    valid: np.ndarray,
    aggregation: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_counts = valid.sum(axis=0)
    positive_counts = np.logical_and(labels > 0, valid).sum(axis=0)
    fractions = np.zeros(labels.shape[1], dtype=np.float32)
    np.divide(
        positive_counts,
        valid_counts,
        out=fractions,
        where=valid_counts > 0,
    )

    if aggregation == 'majority':
        aggregated = fractions > 0.5
    elif aggregation == 'any':
        aggregated = positive_counts > 0
    elif aggregation == 'threshold':
        aggregated = fractions >= threshold
    else:
        raise ValueError(
            f'Unsupported curtain aggregation mode: {aggregation}. '
            'Expected one of "majority", "any", or "threshold".'
        )

    valid_bins = valid_counts > 0
    aggregated &= valid_bins
    return aggregated.astype(np.float32), valid_bins


def rasterize_track_to_dense(
    curtain: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    valid_profile_bins: np.ndarray,
    image_shape: Tuple[int, int],
    aggregation: str = 'majority',
    threshold: float = 0.5,
):
    height_bins = curtain.shape[1]
    height, width = image_shape
    label = np.zeros((height_bins, height, width), dtype=np.float32)
    valid = np.zeros((height_bins, height, width), dtype=bool)
    track_mask = np.zeros((height, width), dtype=bool)

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (
        (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )

    if not np.any(in_bounds):
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_cols = np.empty((0,), dtype=np.int64)
        return label, valid, empty_rows, empty_cols, track_mask

    kept_rows = rows[in_bounds].astype(np.int64)
    kept_cols = cols[in_bounds].astype(np.int64)
    kept_curtain = curtain[in_bounds]
    kept_valid = valid_profile_bins[in_bounds]

    grouped_indices = defaultdict(list)
    for idx, coordinate in enumerate(zip(kept_rows.tolist(), kept_cols.tolist())):
        grouped_indices[coordinate].append(idx)

    for (row, col), profile_indices in grouped_indices.items():
        profile_labels = kept_curtain[profile_indices]
        profile_valid = kept_valid[profile_indices]
        aggregated_labels, aggregated_valid = aggregate_profile_bins(
            profile_labels,
            profile_valid,
            aggregation,
            threshold,
        )
        label[:, row, col] = aggregated_labels
        valid[:, row, col] = aggregated_valid
        track_mask[row, col] = True

    return label, valid, kept_rows, kept_cols, track_mask


def extract_curtain_from_dense(dense, rows, cols):
    if torch.is_tensor(dense):
        row_idx = torch.as_tensor(rows, device=dense.device, dtype=torch.long)
        col_idx = torch.as_tensor(cols, device=dense.device, dtype=torch.long)
        if dense.ndim == 3:
            return dense[:, row_idx, col_idx]
        if dense.ndim == 4:
            return dense[:, :, row_idx, col_idx]
        raise ValueError(f'Expected a 3D or 4D tensor, got shape {tuple(dense.shape)}.')

    dense = np.asarray(dense)
    row_idx = np.asarray(rows, dtype=np.int64)
    col_idx = np.asarray(cols, dtype=np.int64)
    if dense.ndim == 3:
        return dense[:, row_idx, col_idx]
    if dense.ndim == 4:
        return dense[:, :, row_idx, col_idx]
    raise ValueError(f'Expected a 3D or 4D array, got shape {dense.shape}.')


def _prepare_pos_weight(
    pos_weight: Optional[Union[float, Sequence[float], torch.Tensor]],
    device: torch.device,
    num_bins: int,
):
    if pos_weight is None:
        return None

    pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    if pos_weight_tensor.numel() == 1:
        return pos_weight_tensor.reshape(())
    if pos_weight_tensor.numel() != num_bins:
        raise ValueError(
            'pos_weight must be a scalar or have one value per output bin. '
            f'Got {pos_weight_tensor.numel()} values for {num_bins} bins.'
        )
    return pos_weight_tensor.view(num_bins, 1, 1)


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
):
    pos_weight_tensor = _prepare_pos_weight(pos_weight, logits.device, logits.shape[1])
    valid_float = valid.float()

    with torch.autocast(device_type=logits.device.type, enabled=False):
        loss_map = F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            reduction='none',
            pos_weight=pos_weight_tensor,
        )
        weighted_loss = loss_map * valid_float
        valid_count = valid_float.sum()
        loss = weighted_loss.sum() / valid_count.clamp(min=1.0)

    return loss, valid_count


def build_masked_metric_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
):
    flat_logits = logits.detach().float().reshape(-1)
    flat_targets = targets.detach().to(torch.int64).reshape(-1)
    flat_valid = valid.detach().reshape(-1).bool()

    if not flat_valid.any():
        return None, None

    ignored_targets = torch.full_like(flat_targets, ignore_index)
    metric_targets = torch.where(flat_valid, flat_targets, ignored_targets)
    return flat_logits, metric_targets


def update_per_height_iou_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    intersection: torch.Tensor,
    union: torch.Tensor,
    threshold: float = 0.5,
):
    preds = torch.sigmoid(logits.detach().float()) >= threshold
    targets_bool = targets.detach().bool()
    valid_bool = valid.detach().bool()

    intersection += ((preds & targets_bool) & valid_bool).sum(dim=(0, 2, 3)).to(
        intersection.dtype
    )
    union += ((preds | targets_bool) & valid_bool).sum(dim=(0, 2, 3)).to(
        union.dtype
    )


def safe_metric_compute(metric, default: float = 0.0):
    if metric is None:
        return torch.tensor(default, dtype=torch.float32)
    try:
        return metric.compute()
    except RuntimeError:
        return torch.tensor(default, dtype=torch.float32, device=getattr(metric, 'device', None))
