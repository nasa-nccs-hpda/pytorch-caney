import lightning as L
from torch.utils.data import DataLoader

from pytorch_caney.datasets.sharded_dataset import ShardedDataset
from pytorch_caney.transforms.mim_modis_toa import MimTransform


# -----------------------------------------------------------------------------
# SatVisionToaPretrain
# -----------------------------------------------------------------------------
class ModisToaMimDataModule(L.LightningDataModule):
    """NonGeo MODIS TOA Masked-Image-Modeling data module implementation"""

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config,) -> None:
        super().__init__()
        self.config = config
        self.transform = MimTransform(config)
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.train_data_paths = config.DATA.DATA_PATHS
        self.train_data_length = config.DATA.LENGTH
        self.pin_memory = config.DATA.PIN_MEMORY

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = ShardedDataset(
                self.config,
                self.train_data_paths,
                split='train',
                length=self.train_data_length,
                img_size=self.img_size,
                transform=self.transform,
                batch_size=self.batch_size).dataset()

    # -------------------------------------------------------------------------
    # train_dataloader
    # -------------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
