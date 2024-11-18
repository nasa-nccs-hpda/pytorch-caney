from .abi_3dcloud_datamodule import AbiToa3DCloudDataModule
from .modis_toa_mim_datamodule import ModisToaMimDataModule


DATAMODULES = {
    'abitoa3dcloud': AbiToa3DCloudDataModule,
    'modistoamimpretrain': ModisToaMimDataModule,
}


def get_available_datamodules():
    return {name: cls for name, cls in DATAMODULES.items()}
