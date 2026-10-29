import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pytorch_caney.three_d_cloud_track_utils import extract_curtain_from_dense
from pytorch_caney.transforms.abi_radiance_conversion import ConvertABIToReflectanceBT


def create_true_color(chip, rgb_indices):
    red = np.clip(chip[..., rgb_indices[0]] / 100.0, 0, 1)
    green_veggie = np.clip(chip[..., rgb_indices[1]] / 100.0, 0, 1)
    blue = np.clip(chip[..., rgb_indices[2]] / 100.0, 0, 1)

    gamma = 2.2
    red_gamma = np.power(red, 1 / gamma)
    green_gamma = np.power(green_veggie, 1 / gamma)
    blue_gamma = np.power(blue, 1 / gamma)
    green_true = np.clip((0.45 * red_gamma) + (0.1 * green_gamma) + (0.45 * blue_gamma), 0, 1)
    return np.stack([red_gamma, green_true, blue_gamma], axis=-1)


def normalize_ir(channel):
    channel = np.asarray(channel)
    finite = np.isfinite(channel)
    if not finite.any():
        return np.zeros(channel.shape, dtype=np.float32)
    channel_min = channel[finite].min()
    channel_max = channel[finite].max()
    if channel_max == channel_min:
        normalized = np.zeros_like(channel, dtype=np.float32)
        normalized[finite] = 1.0
        return normalized
    normalized = np.zeros_like(channel, dtype=np.float32)
    normalized[finite] = (channel[finite] - channel_min) / (channel_max - channel_min)
    return normalized


def render_chip_for_plotting(selected_chip, config, mean_sza):
    chip = ConvertABIToReflectanceBT()(selected_chip.copy())
    channel_lookup = {
        channel_number: index for index, channel_number in enumerate(config.DATA.INPUT_CHANNELS)
    }
    rgb_indices = [channel_lookup[channel] for channel in config.PLOTTING.TRUE_COLOR_CHANNELS]
    if mean_sza < config.PLOTTING.DAY_NIGHT_SZA_THRESHOLD:
        return create_true_color(chip, rgb_indices), 'True color'

    night_index = channel_lookup[config.PLOTTING.NIGHT_IR_CHANNEL]
    return normalize_ir(chip[..., night_index]), f'ABI ch {config.PLOTTING.NIGHT_IR_CHANNEL}'


def plot_track_predictions(path, samples, predictions, config):
    pdf_plot_obj = PdfPages(path)

    for sample, prediction in zip(samples, predictions):
        rows = sample['track_rows']
        cols = sample['track_cols']
        curtain_truth = sample['curtain'].T
        curtain_valid = sample['curtain_valid'].T
        curtain_pred = extract_curtain_from_dense(prediction, rows, cols)
        curtain_pred = 1.0 / (1.0 + np.exp(-np.asarray(curtain_pred)))
        curtain_diff = np.where(curtain_valid, curtain_pred - curtain_truth, np.nan)

        chip_display, chip_title = render_chip_for_plotting(
            sample['selected_chip'],
            config,
            sample['mean_sza'],
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_chip, ax_truth, ax_pred, ax_diff = axes.flatten()

        if chip_display.ndim == 3:
            ax_chip.imshow(chip_display)
        else:
            ax_chip.imshow(chip_display, cmap='inferno')
        ax_chip.scatter(cols, rows, c='cyan', s=8)
        ax_chip.set_title(f'{chip_title} with EarthCARE track')

        ax_truth.imshow(curtain_truth, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax_truth.set_title('EarthCARE curtain')

        ax_pred.imshow(curtain_pred, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax_pred.set_title('Predicted curtain')

        diff_image = ax_diff.imshow(curtain_diff, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax_diff.set_title('Prediction - truth')
        fig.colorbar(diff_image, ax=ax_diff, fraction=0.046, pad=0.04)

        for axis in axes.flatten():
            axis.set_xlabel('Track / column')
            axis.set_ylabel('Height bin')

        fig.tight_layout()
        pdf_plot_obj.savefig(fig)
        plt.close(fig)

    pdf_plot_obj.close()
