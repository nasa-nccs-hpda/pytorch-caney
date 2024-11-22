from torch.utils.data import DataLoader
import lightning as L

from pytorch_caney.datasets.abi_3dcloud_dataset import AbiToa3DCloudDataset
from pytorch_caney.transforms.abi_toa import AbiToaTransform


# -----------------------------------------------------------------------------
# AbiToa3DCloudDataModule
# -----------------------------------------------------------------------------
class AbiToa3DCloudDataModule(L.LightningDataModule):
    """NonGeo ABI TOA 3D cloud data module implementation"""

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(
                self,
                config,
            ) -> None:
        super().__init__()
        self.config = config
        self.transform = AbiToaTransform(config.DATA.IMG_SIZE)
        print(self.transform)
        self.train_data_paths = config.DATA.DATA_PATHS
        self.test_data_paths = config.DATA.TEST_DATA_PATHS
        self.batch_size = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = AbiToa3DCloudDataset(
                self.config,
                self.train_data_paths,
                self.transform,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = AbiToa3DCloudDataset(
                self.config,
                self.test_data_paths,
                self.transform,
            )
        if stage in ["test"]:
            self.test_dataset = AbiToa3DCloudDataset(
                self.config,
                self.test_data_paths,
                self.transform,
            )

    # -------------------------------------------------------------------------
    # train_dataloader
    # -------------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    # -------------------------------------------------------------------------
    # val_dataloader
    # -------------------------------------------------------------------------
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    # -------------------------------------------------------------------------
    # test_dataloader
    # -------------------------------------------------------------------------
    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
