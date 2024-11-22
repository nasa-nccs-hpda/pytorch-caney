import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Use a PL data module
_C.DATA.DATAMODULE = True
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path(s) to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATHS = ['']
# Path(s) to the validation/test dataset
_C.DATA.TEST_DATA_PATHS = ['']
# Path(s) to dataset masks
_C.DATA.MASK_PATHS = ['']
# Path to validation numpy dataset
_C.DATA.VALIDATION_PATH = ''
# Dataset name
_C.DATA.DATASET = 'MODIS'
# Input image size
_C.DATA.IMG_SIZE = 224
# Dataset length (for datasets where len cannot be used)
_C.DATA.LENGTH = 1920000
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swinv2'
# Encoder type for fine-tuning
_C.MODEL.ENCODER = ''
# Decoder type for fine-tuning
_C.MODEL.DECODER = ''
# Model name
_C.MODEL.NAME = 'swinv2_base_patch4_window7_224'
# Pretrained weight from checkpoint, could be from previous pre-training
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 17
# Number of channels the input image has
_C.MODEL.IN_CHANS = 3
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWINV2.NORM_PERIOD = 0
_C.MODEL.SWINV2.NORM_STAGE = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.NAME = 'tversky'
_C.LOSS.MODE = 'multiclass'
_C.LOSS.CLASSES = None
_C.LOSS.LOG = False
_C.LOSS.LOGITS = True
_C.LOSS.SMOOTH = 0.0
_C.LOSS.IGNORE_INDEX = None
_C.LOSS.EPS = 1e-7
_C.LOSS.ALPHA = 0.5
_C.LOSS.BETA = 0.5
_C.LOSS.GAMMA = 1.0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.ACCELERATOR = 'gpu'
_C.TRAIN.STRATEGY = 'deepspeed'
_C.TRAIN.LIMIT_TRAIN_BATCHES = True
_C.TRAIN.NUM_TRAIN_BATCHES = None
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WARMUP_STEPS = 200
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
# OneCycle LR Scheduler max LR percentage
_C.TRAIN.LR_SCHEDULER.CYCLE_PERCENTAGE = 0.3

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# Tensorboard settings
_C.TENSORBOARD = CN()
_C.TENSORBOARD.WRITER_DIR = '.'

# DeepSpeed configuration settings
_C.DEEPSPEED = CN()
_C.DEEPSPEED.STAGE = 2
_C.DEEPSPEED.REDUCE_BUCKET_SIZE = 5e8
_C.DEEPSPEED.ALLGATHER_BUCKET_SIZE = 5e8
_C.DEEPSPEED.CONTIGUOUS_GRADIENTS = True
_C.DEEPSPEED.OVERLAP_COMM = True


# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Whether to enable pytorch amp, overwritten by command line argument
_C.PRECISION = '32'
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = '.'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'pt-caney-default-tag'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Frequency for running validation step
_C.VALIDATION_FREQ = 1
# Fixed random seed
_C.SEED = 42
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Pipeline
_C.PIPELINE = 'satvisiontoapretrain'
# Data module
_C.DATAMODULE = 'abitoa3dcloud'
# Fast dev run
_C.FAST_DEV_RUN = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_paths'):
        config.DATA.DATA_PATHS = args.data_paths
    if _check_args('validation_path'):
        config.DATA.VALIDATION_PATH = args.validation_path
    if _check_args('dataset'):
        config.DATA.DATASET = args.dataset
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp
    if _check_args('tensorboard_dir'):
        config.TENSORBOARD.WRITER_DIR = args.tensorboard_dir

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
