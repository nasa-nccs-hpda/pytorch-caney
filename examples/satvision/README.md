# SatVision Examples

The following is an example on how to run SatVision finetune. This is only an example and does not limit other decoder possibilities, or other ways of dealing with the encoder.

## SatVision Finetune Land Cover Five Class

The script run_satvision_finetune_lc_fiveclass.sh has an example on how to run the finetuning of a 5 class land cover model using a simple UNet architecture. The dependencies of this model are as follow:

- finetune.py script (pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py): this script has the basics for training the finetuning model. Below you will find an example of this script:

```bash
export PYTHONPATH=$PWD:pytorch-caney
export NGPUS=8

torchrun --nproc_per_node $NGPUS \
    pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py \
	--cfg finetune_satvision_base_landcover5class_192_window12_100ep.yaml \
	--pretrained /explore/nobackup/people/cssprad1/projects/satnet/code/development/masked_image_modeling/development/models/simmim_satnet_pretrain_pretrain/simmim_pretrain__satnet_swinv2_base__img192_window12__800ep_v3_no_norm/ckpt_epoch_800.pth \
    --dataset MODISLC9 \
	--data-paths /explore/nobackup/projects/ilab/data/satvision/finetuning/h18v04/labels_9classes_224 \
	--batch-size 4 \
	--output /explore/nobackup/people/cssprad1/projects/satnet/code/development/cleanup/finetune/models \
	--enable-amp
```

From these parameters note that:

- the pretrained model path is given by --pretrained
- the data paths is given by --data-paths and is expecting a directory whose internal structure is one for images and one from labels, but this can be modified if both input and target files are stored in the same file
- the dataloader is simply called from the script using the --dataset option, which is simply calling build_finetune_dataloaders
from pytorch-caney

These is simply a guide script on how to run a finetuning pipeline. If you want to get additional insights on how to build other
types of decoders, the build_model function from pytorch_caney/models/build.py has additional details on how to combine the different
encoder and decoders.