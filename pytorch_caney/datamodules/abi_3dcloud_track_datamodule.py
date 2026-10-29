import lightning as L
from torch.utils.data import DataLoader

from pytorch_caney.datasets.abi_3dcloud_track_dataset import AbiToa3DCloudTrackDataset
from pytorch_caney.transforms.abi_toa import AbiToaTransform


class AbiToa3DCloudTrackDataModule(L.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transform = AbiToaTransform(config.DATA.IMG_SIZE)
        self.train_data_paths = config.DATA.DATA_PATHS
        self.test_data_paths = config.DATA.TEST_DATA_PATHS
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.pin_memory = config.DATA.PIN_MEMORY

    def setup(self, stage: str) -> None:
        if stage in ['fit']:
            self.train_dataset = AbiToa3DCloudTrackDataset(
                self.config,
                self.train_data_paths,
                self.transform,
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = AbiToa3DCloudTrackDataset(
                self.config,
                self.test_data_paths,
                self.transform,
            )
        if stage in ['test']:
            self.test_dataset = AbiToa3DCloudTrackDataset(
                self.config,
                self.test_data_paths,
                self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
