## USER GUIDE - Pipeline 2. 3D Cloud Retrieval

**_NOTE: The Getting Started section must be completed [first.](requirements/README.md)_**

See _4.2 3D Cloud Retrieval Downstream Task_ (https://arxiv.org/pdf/2411.17000) for prediction details.

## Input
The runtime script, which  predicts a 3D cloud vertical structure, specifies the following default values.  Modify ```configs/3dcloud_retrieval.yaml``` directly to change these paths. 

### Key Configuration Parameters 
| Command-line-argument | Description                       | Default  |
| --------------------- |:----------------------------------|:---------|
| `ENCODER`          | reconstructor | satvision |
| `DECODER`          | deconstructor | fcn |
| `PRETRAINED`         | Path to training model settings   | ./satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt |
| `DATA_PATHS`           | Path to cropped MODIS input images           | /explore/nobackup/projects/ilab/data/satvision-toa/3dcloud.data/abi_chips/handpicked/ |
| `EPOCHS`           | # of ML iterations           | 50 |
| `NAME`           | Model name           | 3dcloud-retrieval |
| `TAG`           | # instance of output        | 3dcloud-retrieval |

Annotated commands here, actual session with expected results follows:
| Description                       | Syntax  |
| ----------------------------------|:---------|
| Run script  | python pytorch-caney/pytorch_caney/ptc_cli.py --config-path ./pytorch-caney/configs/3dcloud_retrieval.yaml |


### _Sample Session - 3D Cloud Retrieval_ 

```bash
(base) <user>@gpu002:/explore/nobackup/projects/ilab/projects/Satvision$ time singularity exec --nv -B /explore/nobackup/projects/ilab/projects/Satvision,/explore/nobackup/projects/ilab/data/satvision-toa /explore/nobackup/projects/ilab/containers/pytorch-caney-container python ./pytorch-caney/pytorch_caney/ptc_cli.py --config-path ./pytorch-caney/configs/3dcloud_retrieval.yaml
WARNING: underlay of /etc/localtime required more than 50 (117) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (616) bind mounts
13:4: not a valid test operator: (
13:4: not a valid test operator: 570.86.15
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
There was a problem when trying to write in your cache folder (/home/gtamkin/.cache/huggingface/hub). You should set the environment variable TRANSFORMERS_CACHE to a writable directory.
/usr/local/lib/python3.10/dist-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
[2025-02-27 09:36:41,706] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The cache directory for DeepSpeed Triton autotune, /home/gtamkin/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
=> merge config from /explore/nobackup/projects/ilab/projects/Satvision/pytorch-caney/configs/3dcloudtask_swinv2_satvision_giant_test_gt.yaml
Output directory: ./3dcloud-svtoa-finetune-giant/3dcloud_task_swinv2_g_satvision_128_scaled_bt_minmax
Full config saved to ./3dcloud-svtoa-finetune-giant/3dcloud_task_swinv2_g_satvision_128_scaled_bt_minmax/3dcloud_task_swinv2_g_satvision_128_scaled_bt_minmax.config.json
AMP_ENABLE: true
BASE:
- ''
DATA:
  BATCH_SIZE: 32
  DATAMODULE: true
  DATASET: MODIS
  DATA_PATHS:
  - /explore/nobackup/projects/ilab/data/satvision-toa/3dcloud.data/abi_chips/handpicked/
  IMG_SIZE: 128
  INTERPOLATION: bicubic
  LENGTH: 1920000
  MASK_PATCH_SIZE: 32
  MASK_PATHS:
  - ''
  MASK_RATIO: 0.6
  NUM_WORKERS: 8
  PIN_MEMORY: true
  TEST_DATA_PATHS:
  - /explore/nobackup/projects/ilab/data/satvision-toa/3dcloud.data/abiChipsNew/
  VALIDATION_PATH: ''
DATAMODULE: abitoa3dcloud
DEEPSPEED:
  ALLGATHER_BUCKET_SIZE: 500000000.0
  CONTIGUOUS_GRADIENTS: true
  OVERLAP_COMM: true
  REDUCE_BUCKET_SIZE: 500000000.0
  STAGE: 2
EVAL_MODE: false
FAST_DEV_RUN: false
LOSS:
  ALPHA: 0.5
  BETA: 0.5
  CLASSES: null
  EPS: 1.0e-07
  GAMMA: 1.0
  IGNORE_INDEX: null
  LOG: false
  LOGITS: true
  MODE: multiclass
  NAME: bce
  SMOOTH: 0.0
MODEL:
  DECODER: fcn
  DROP_PATH_RATE: 0.1
  DROP_RATE: 0.0
  ENCODER: satvision
  IN_CHANS: 14
  NAME: 3dcloud-svtoa-finetune-giant
  NUM_CLASSES: 17
  PRETRAINED: /explore/nobackup/projects/ilab/projects/Satvision/input/satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt
  RESUME: ''
  SWINV2:
    APE: false
    DEPTHS:
    - 2
    - 2
    - 42
    - 2
    EMBED_DIM: 512
    IN_CHANS: 14
    MLP_RATIO: 4.0
    NORM_PERIOD: 6
    NORM_STAGE: false
    NUM_HEADS:
    - 16
    - 32
    - 64
    - 128
    PATCH_NORM: true
    PATCH_SIZE: 4
    PRETRAINED_WINDOW_SIZES:
    - 0
    - 0
    - 0
    - 0
    QKV_BIAS: true
    WINDOW_SIZE: 8
  TYPE: swinv2
OUTPUT: .
PIPELINE: 3dcloud
PRECISION: '32'
PRINT_FREQ: 10
SAVE_FREQ: 50
SEED: 42
TAG: 3dcloud_task_swinv2_g_satvision_128_scaled_bt_minmax
TENSORBOARD:
  WRITER_DIR: .
TEST:
  CROP: true
TRAIN:
  ACCELERATOR: gpu
  ACCUMULATION_STEPS: 1
  AUTO_RESUME: true
  BASE_LR: 0.0003
  CLIP_GRAD: 5.0
  EPOCHS: 50
  LAYER_DECAY: 1.0
  LIMIT_TRAIN_BATCHES: true
  LR_SCHEDULER:
    CYCLE_PERCENTAGE: 0.3
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
    GAMMA: 0.1
    MULTISTEPS:
    - 700
    NAME: multistep
  MIN_LR: 0.0002
  NUM_TRAIN_BATCHES: null
  OPTIMIZER:
    BETAS:
    - 0.9
    - 0.999
    EPS: 1.0e-08
    MOMENTUM: 0.9
    NAME: adamw
  START_EPOCH: 0
  STRATEGY: deepspeed
  USE_CHECKPOINT: true
  WARMUP_EPOCHS: 10
  WARMUP_LR: 0.0001
  WARMUP_STEPS: 200
  WEIGHT_DECAY: 0.05
VALIDATION_FREQ: 20

Training
Available pipelines: {'satvisiontoapretrain': <class 'pytorch_caney.pipelines.satvision_toa_pretrain_pipeline.SatVisionToaPretrain'>, '3dcloud': <class 'pytorch_caney.pipelines.three_d_cloud_pipeline.ThreeDCloudTask'>}
Using <class 'pytorch_caney.pipelines.three_d_cloud_pipeline.ThreeDCloudTask'>
{'fcn': <class 'pytorch_caney.models.encoders.fcn_encoder.FcnEncoder'>, 'swinv2': <class 'pytorch_caney.models.encoders.swinv2.SwinTransformerV2'>, 'satvision': <class 'pytorch_caney.models.encoders.satvision.SatVision'>}
{'fcn': <class 'pytorch_caney.models.decoders.fcn_decoder.FcnDecoder'>}
{'segmentation_head': <class 'pytorch_caney.models.heads.segmentation_head.SegmentationHead'>}
/usr/local/lib/python3.10/dist-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Detect pre-trained model, remove [encoder.] prefix.
_IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=['mask_token', 'layers.2.blocks.5.norm3.weight', 'layers.2.blocks.5.norm3.bias', 'layers.2.blocks.11.norm3.weight', 'layers.2.blocks.11.norm3.bias', 'layers.2.blocks.17.norm3.weight', 'layers.2.blocks.17.norm3.bias', 'layers.2.blocks.23.norm3.weight', 'layers.2.blocks.23.norm3.bias', 'layers.2.blocks.29.norm3.weight', 'layers.2.blocks.29.norm3.bias', 'layers.2.blocks.35.norm3.weight', 'layers.2.blocks.35.norm3.bias', 'layers.2.blocks.41.norm3.weight', 'layers.2.blocks.41.norm3.bias'])
>>>>>>> loaded successfully '/explore/nobackup/projects/ilab/projects/Satvision/input/satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt'
. . . . . 

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0003
    maximize: False
    weight_decay: 0.0
)

   | Name              | Type               | Params | Mode 
------------------------------------------------------------------
0  | encoder           | SatVision          | 2.6 B  | train
1  | decoder           | FcnDecoder         | 86.5 M | train
2  | segmentation_head | SegmentationHead   | 577    | train
3  | model             | Sequential         | 2.7 B  | train
4  | criterion         | BCEWithLogitsLoss  | 0      | train
5  | train_iou         | BinaryJaccardIndex | 0      | train
6  | val_iou           | BinaryJaccardIndex | 0      | train
7  | train_loss_avg    | MeanMetric         | 0      | train
8  | val_loss_avg      | MeanMetric         | 0      | train
9  | train_iou_avg     | MeanMetric         | 0      | train
10 | val_iou_avg       | MeanMetric         | 0      | train
------------------------------------------------------------------
2.7 B     Trainable params
0         Non-trainable params
2.7 B     Total params
10,726.302Total estimated model params size (MB)
1012      Modules in train mode
0         Modules in eval mode
Sanity Checking: |                                                                                                                                                                                                                                               | 0/? [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/lightning/pytorch/utilities/data.py:106: Total length of `DataLoader` across ranks is zero. Please make sure this was your intention.
/usr/local/lib/python3.10/dist-packages/lightning/pytorch/loops/fit_loop.py:310: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=10). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.        
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:06<00:00,  0.14it/s, v_num=6, train_loss_step=0.685, train_iou_step=0.000]/usr/local/lib/python3.10/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:434: It is recommended to use `self.log('train_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
/usr/local/lib/python3.10/dist-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:434: It is recommended to use `self.log('train_iou', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
Epoch 49: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  0.93it/s, v_num=6, train_loss_step=0.304, train_iou_step=0.000, train_loss_epoch=0.304, train_iou_epoch=0.000]`Trainer.fit` stopped: `max_epochs=50` reached.
Epoch 49: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:50<00:00,  0.02it/s, v_num=6, train_loss_step=0.304, train_iou_step=0.000, train_loss_epoch=0.304, train_iou_epoch=0.000]

real	41m26.495s
user	11m2.219s
sys	12m11.982s

```

## Output

The results, logs, and model checkpoints are saved in a directory specified by ```<output-dir>/<model-name>/<tag>```.  For example:

``` bash
(base) <user>@ilab213:/explore/nobackup/projects/ilab/projects/Satvision$ ls -aRt 3dcloud-retrieval/
3dcloud-retrieval/:
3dcloud-retrieval  .  ..

3dcloud-retrieval/3dcloud-retrieval:
lightning_logs  3dcloud-retrieval.config.json  .  ..

3dcloud-retrieval/3dcloud-retrieval/lightning_logs:
.  version_1  ..

3dcloud-retrieval/3dcloud-retrieval/lightning_logs/version_1:
..  events.out.tfevents.1740944483.gpu003.2139529.0  checkpoints  .  hparams.yaml

3dcloud-retrieval/3dcloud-retrieval/lightning_logs/version_1/checkpoints:
 .  'epoch=49-step=50.ckpt'   ..

'3dcloud-retrieval/3dcloud-retrieval/lightning_logs/version_1/checkpoints/epoch=49-step=50.ckpt':
..  latest  .  zero_to_fp32.py  checkpoint

'3dcloud-retrieval/3dcloud-retrieval/lightning_logs/version_1/checkpoints/epoch=49-step=50.ckpt/checkpoint':
zero_pp_rank_1_mp_rank_00_optim_states.pt  ..  zero_pp_rank_0_mp_rank_00_optim_states.pt  zero_pp_rank_2_mp_rank_00_optim_states.pt  zero_pp_rank_3_mp_rank_00_optim_states.pt  .  mp_rank_00_model_states.pt

```
