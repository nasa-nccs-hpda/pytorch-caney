from .satvision_toa_pretrain_pipeline import SatVisionToaPretrain
from .three_d_cloud_pipeline import ThreeDCloudTask
from .three_d_cloud_track_pipeline import ThreeDCloudTrackTask


PIPELINES = {
    'satvisiontoapretrain': SatVisionToaPretrain,
    '3dcloud': ThreeDCloudTask,
    '3dcloud_track': ThreeDCloudTrackTask,
}


def get_available_pipelines():
    return {name: cls for name, cls in PIPELINES.items()}
