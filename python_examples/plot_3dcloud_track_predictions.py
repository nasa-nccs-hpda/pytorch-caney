import argparse
from pathlib import Path

import numpy as np
import torch

from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.datasets.abi_3dcloud_track_dataset import AbiToa3DCloudTrackDataset
from pytorch_caney.plotting.three_d_cloud_track import plot_track_predictions
from pytorch_caney.pipelines import PIPELINES
from pytorch_caney.transforms.abi_toa import AbiToaTransform


def main(config, checkpoint_path: str, output_path: str, num_samples: int):
    transform = AbiToaTransform(config.DATA.IMG_SIZE)
    dataset = AbiToa3DCloudTrackDataset(config, config.DATA.TEST_DATA_PATHS, transform=transform)
    pipeline = PIPELINES[config.PIPELINE].load_from_checkpoint(checkpoint_path, config=config)
    pipeline.eval()
    pipeline.cpu()

    samples = []
    predictions = []
    for index in range(min(num_samples, len(dataset))):
        sample = dataset.load_track_sample(index)
        with torch.no_grad():
            image = sample['image'].unsqueeze(0)
            prediction = pipeline(image).squeeze(0).cpu().numpy()

        samples.append(sample)
        predictions.append(prediction)

    plot_track_predictions(output_path, samples, predictions, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=8)
    args = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, args.config_path)

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    main(config, args.checkpoint_path, args.output_path, args.num_samples)
