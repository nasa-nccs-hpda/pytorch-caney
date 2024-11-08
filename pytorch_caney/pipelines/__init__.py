from .satvision_toa_pretrain_pipeline import SatVisionToaPretrain
from .three_d_cloud_pipeline import ThreeDCloudTask 

PIPELINES = {
    'satvisiontoapretrain': SatVisionToaPretrain,
    '3dcloud': ThreeDCloudTask
}

def get_available_pipelines():
    return {name: cls for name, cls in PIPELINES.items()}
