from .abi_3dcloud_datamodule import AbiToa3DCloudDataModule

DATAMODULES = {
    'abitoa3dcloud': AbiToa3DCloudDataModule,
}

def get_available_datamodules():
    return {name: cls for name, cls in DATAMODULES.items()}