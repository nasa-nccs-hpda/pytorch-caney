import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rioxarray as rxr
import torch

from torchgeo.datasets import NonGeoDataset

from pytorch_caney.three_d_cloud_track_utils import (
    apply_track_pixel_offsets,
    build_profile_valid_mask,
    channel_numbers_to_indices,
    rasterize_track_to_dense,
)


class AbiToa3DCloudTrackDataset(NonGeoDataset):
    def __init__(
        self,
        config,
        data_paths: list,
        transform=None,
        split: str = "train",
        audit_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.data_paths = data_paths
        self.transform = transform
        self.split = split
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        self.img_size = config.DATA.IMG_SIZE
        self.output_bins = config.MODEL.OUTPUT_BINS
        self.channel_numbers = list(config.DATA.INPUT_CHANNELS)
        self.channel_indices = channel_numbers_to_indices(self.channel_numbers)
        self.solar_zenith_index = int(config.DATA.SOLAR_ZENITH_CHANNEL) - 1
        self.curtain_key = config.DATA.CURTAIN_KEY
        self.curtain_valid_key = config.DATA.CURTAIN_VALID_KEY
        self.curtain_height_key = config.DATA.CURTAIN_HEIGHT_KEY
        self.track_row_key = config.DATA.TRACK_ROW_KEY
        self.track_col_key = config.DATA.TRACK_COL_KEY
        self.curtain_fill_value = config.DATA.CURTAIN_FILL_VALUE
        self.exclude_top_bins = config.DATA.CURTAIN_EXCLUDE_TOP_BINS
        self.exclude_bottom_bins = config.DATA.CURTAIN_EXCLUDE_BOTTOM_BINS
        self.aggregation = config.DATA.CURTAIN_AGGREGATION
        self.aggregation_threshold = config.DATA.CURTAIN_AGGREGATION_THRESHOLD
        self.return_metadata = config.DATA.RETURN_METADATA

        if self.channel_indices and len(self.channel_indices) != config.MODEL.IN_CHANS:
            raise ValueError(
                "MODEL.IN_CHANS must match the selected input channels. "
                f"Got {config.MODEL.IN_CHANS} and {len(self.channel_indices)}."
            )

        self.image_list = []
        for image_mask_path in self.data_paths:
            self.image_list.extend(self.get_filenames(image_mask_path))

        self.num_candidate_files = len(self.image_list)
        self.image_list, self.skipped_nonfinite_files = self._filter_nonfinite_input_files(
            self.image_list
        )
        self.audit_file_path = self._write_nonfinite_audit_file()
        skipped_count = len(self.skipped_nonfinite_files)
        print(
            f"[{self.split}] filtered {skipped_count} / {self.num_candidate_files} files "
            "with non-finite selected input channels"
        )
        if self.audit_file_path is not None:
            print(f"[{self.split}] audit file written to {self.audit_file_path}")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        sample = self.load_track_sample(index, include_metadata=self.return_metadata)
        if self.return_metadata:
            return sample["image"], sample["label"], sample["valid"], sample["metadata"]
        return sample["image"], sample["label"], sample["valid"]

    def load_track_sample(
        self, index: int, include_metadata: bool = False
    ) -> Dict[str, Any]:
        path = self.image_list[index]
        npz_array = self._load_file(path)
        payload = npz_array["data"].item()

        raw_chip = np.array(npz_array["chip"], copy=True)
        self._assert_chip_size(raw_chip, path)
        selected_chip = self._select_input_channels(raw_chip)
        self._assert_selected_chip_finite(selected_chip, path)

        # Any future spatial augmentation must transform the image, label, and
        # validity mask identically to keep the dense supervision aligned.
        image = selected_chip.copy()
        if self.transform is not None:
            image = self.transform(image)

        self._assert_transformed_image_finite(image, path)

        curtain = np.asarray(payload[self.curtain_key])
        if curtain.ndim != 2:
            raise ValueError(
                f"Expected curtain array with shape (track_length, bins), got {curtain.shape}."
            )

        if curtain.shape[1] != self.output_bins:
            raise ValueError(
                "Curtain bins must match MODEL.OUTPUT_BINS. "
                f"Got {curtain.shape[1]} and {self.output_bins}."
            )

        rows = np.rint(np.asarray(payload[self.track_row_key])).astype(np.int64)
        cols = np.rint(np.asarray(payload[self.track_col_key])).astype(np.int64)
        if rows.shape[0] != curtain.shape[0] or cols.shape[0] != curtain.shape[0]:
            raise ValueError(
                "Track row/col indices must have one entry per curtain profile. "
                f"Got rows={rows.shape}, cols={cols.shape}, curtain={curtain.shape}."
            )

        validity_source = payload.get(self.curtain_valid_key)
        curtain_valid = build_profile_valid_mask(
            curtain,
            validity_source=np.asarray(validity_source)
            if validity_source is not None
            else None,
            fill_value=self.curtain_fill_value,
            exclude_top_bins=self.exclude_top_bins,
            exclude_bottom_bins=self.exclude_bottom_bins,
        )

        rows, cols = apply_track_pixel_offsets(rows, cols, payload, self.config)
        in_bounds = (
            (rows >= 0) & (rows < self.img_size) & (cols >= 0) & (cols < self.img_size)
        )
        label, valid, track_rows, track_cols, track_mask = rasterize_track_to_dense(
            curtain.astype(np.float32),
            rows,
            cols,
            curtain_valid,
            image_shape=(self.img_size, self.img_size),
            aggregation=self.aggregation,
            threshold=self.aggregation_threshold,
        )

        heights = np.asarray(payload[self.curtain_height_key], dtype=np.float32)
        if heights.ndim != 2 or heights.shape[1] != self.output_bins:
            raise ValueError(
                "Curtain height grid must have shape (track_length, MODEL.OUTPUT_BINS). "
                f"Got {heights.shape}. "
            )

        mean_sza = np.nan
        if self.solar_zenith_index < raw_chip.shape[-1]:
            mean_sza = np.float32(np.nanmean(raw_chip[..., self.solar_zenith_index]))

        sample = {
            "image": image,
            "label": label.astype(np.float32),
            "valid": valid.astype(bool),
            "raw_chip": raw_chip,
            "selected_chip": selected_chip,
            "track_rows": track_rows,
            "track_cols": track_cols,
            "track_mask": track_mask,
            "curtain": curtain[in_bounds].astype(np.float32),
            "curtain_valid": curtain_valid[in_bounds].astype(bool),
            "heights": heights[0].astype(np.float32)
            if heights.shape[0] > 0
            else np.zeros((self.output_bins,), dtype=np.float32),
            "mean_sza": np.float32(mean_sza),
            "sample_path": path,
        }

        if include_metadata:
            sample["metadata"] = self._build_metadata(sample)

        return sample

    def _build_metadata(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        max_track_length = self.img_size * self.img_size
        track_rows = np.full((max_track_length,), -1, dtype=np.int16)
        track_cols = np.full((max_track_length,), -1, dtype=np.int16)
        actual_length = min(len(sample["track_rows"]), max_track_length)
        if actual_length:
            track_rows[:actual_length] = sample["track_rows"][:actual_length].astype(
                np.int16
            )
            track_cols[:actual_length] = sample["track_cols"][:actual_length].astype(
                np.int16
            )

        return {
            "track_rows": track_rows,
            "track_cols": track_cols,
            "track_length": np.int32(actual_length),
            "track_mask": sample["track_mask"].astype(bool),
            "mean_sza": np.float32(sample["mean_sza"]),
        }

    def _assert_chip_size(self, chip: np.ndarray, path: str) -> None:
        expected_shape = (self.img_size, self.img_size)
        if chip.shape[:2] != expected_shape:
            raise ValueError(
                "Track rasterization assumes chip size already matches IMG_SIZE; "
                "otherwise AbiToaTransform.Resize would desynchronize the image and "
                f"labels. Expected {expected_shape}, got {chip.shape[:2]} for {path}."
            )

    def _select_input_channels(self, chip: np.ndarray) -> np.ndarray:
        if not self.channel_indices:
            return chip

        if max(self.channel_indices) >= chip.shape[-1]:
            raise ValueError(
                "Selected input channel exceeds chip depth. "
                f"Got max index {max(self.channel_indices)} for chip with {chip.shape[-1]} channels."
            )

        return chip[..., self.channel_indices]

    def _assert_selected_chip_finite(self, chip: np.ndarray, path: str) -> None:
        if np.isfinite(chip).all():
            return

        nan_count = int(np.isnan(chip).sum())
        inf_count = int(np.isinf(chip).sum())
        raise ValueError(
            f"Non-finite selected input channels remain for {path}: "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )

    def _assert_transformed_image_finite(self, image, path: str) -> None:
        if torch.is_tensor(image):
            is_finite = torch.isfinite(image)
            if is_finite.all():
                return

            nan_count = int(torch.isnan(image).sum().item())
            inf_count = int(torch.isinf(image).sum().item())
            raise ValueError(
                f"Non-finite transformed image for {path}: "
                f"nan_count={nan_count}, inf_count={inf_count}"
            )

        image_array = np.asarray(image)
        if np.isfinite(image_array).all():
            return

        nan_count = int(np.isnan(image_array).sum())
        inf_count = int(np.isinf(image_array).sum())
        raise ValueError(
            f"Non-finite transformed image for {path}: "
            f"nan_count={nan_count}, inf_count={inf_count}"
        )

    def _filter_nonfinite_input_files(self, image_list: list[str]):
        kept_files = []
        skipped_files = []
        for path in image_list:
            raw_chip = np.array(self._load_file(path)["chip"], copy=False)
            self._assert_chip_size(raw_chip, path)
            selected_chip = self._select_input_channels(raw_chip)
            if np.isfinite(selected_chip).all():
                kept_files.append(path)
                continue

            skipped_files.append(self._get_nonfinite_channel_info(path, selected_chip))

        return kept_files, skipped_files

    def _get_nonfinite_channel_info(self, path: str, selected_chip: np.ndarray) -> Dict[str, Any]:
        nonfinite_mask = ~np.isfinite(selected_chip)
        per_channel_nonfinite = nonfinite_mask.sum(axis=(0, 1))
        selected_channel_indices = [
            int(channel_index)
            for channel_index, count in enumerate(per_channel_nonfinite.tolist())
            if count > 0
        ]
        abi_channel_numbers = [int(self.channel_numbers[channel_index]) for channel_index in selected_channel_indices]
        all_nonfinite_selected_channels = [
            int(channel_index)
            for channel_index in selected_channel_indices
            if nonfinite_mask[:, :, channel_index].all()
        ]
        return {
            "sample_path": path,
            "nonfinite_count": int(nonfinite_mask.sum()),
            "selected_channel_indices": selected_channel_indices,
            "abi_channel_numbers": abi_channel_numbers,
            "all_nonfinite_selected_channels": all_nonfinite_selected_channels,
        }

    def _write_nonfinite_audit_file(self) -> Optional[Path]:
        if self.audit_dir is None:
            return None

        self.audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file_path = self.audit_dir / f"skipped_nonfinite_input_chips.{self.split}.txt"
        header = (
            "sample_path\tnonfinite_count\tselected_channel_indices\t"
            "abi_channel_numbers\tall_nonfinite_selected_channels\n"
        )
        with open(audit_file_path, "w") as file_obj:
            file_obj.write(header)
            for entry in self.skipped_nonfinite_files:
                file_obj.write(
                    f"{entry['sample_path']}\t{entry['nonfinite_count']}\t"
                    f"{entry['selected_channel_indices']}\t{entry['abi_channel_numbers']}\t"
                    f"{entry['all_nonfinite_selected_channels']}\n"
                )

        return audit_file_path

    def _load_file(self, path: Path):
        suffix = Path(path).suffix.lower()
        if suffix in [".npy", ".npz"]:
            return np.load(path, allow_pickle=True)
        if suffix == ".tif":
            return rxr.open_rasterio(path)
        raise RuntimeError(
            f"Non-recognized dataset format: {suffix}. Expects .npy, .npz, or .tif"
        )

    def get_filenames(self, path):
        files_list = []
        for filename in sorted(os.listdir(path)):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath) and Path(filepath).suffix.lower() in [
                ".npy",
                ".npz",
                ".tif",
            ]:
                files_list.append(filepath)
        return files_list
