from .satvision_toa_pretrain_pipeline import SatVisionToaPretrain

PIPELINES = {
    'satvisiontoapretrain': SatVisionToaPretrain,
}

def get_available_pipelines():
    return {name: cls for name, cls in PIPELINES.items()}
