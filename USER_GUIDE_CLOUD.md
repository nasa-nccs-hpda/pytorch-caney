## USER GUIDE - Image Reconstruction with Pretrained Model

## Pipeline 1.  Image Reconstruction

See Section _4.1 Image Reconstruction_ (https://arxiv.org/pdf/2411.17000) for reconstruction performance measurements.
* [Run Image Reconstruction with Pretrained Model](./USER_GUIDE_IMAGE.md)

### Sample Session - Image Reconstruction 

```bash
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision/pytorch-caney$ module load singularity
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision/pytorch-caney$ export PYTHONPATH=$PWD:$PWD/pytorch-caney
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision/pytorch-caney$ singularity exec --nv -B /explore/nobackup/projects/ilab/projects /explore/nobackup/projects/ilab/containers/pytorch-caney-container python tests/image_reconstruction.py 
WARNING: underlay of /etc/localtime required more than 50 (117) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (616) bind mounts
13:4: not a valid test operator: (
13:4: not a valid test operator: 570.86.15
There was a problem when trying to write in your cache folder (/home/gtamkin/.cache/huggingface/hub). You should set the environment variable TRANSFORMERS_CACHE to a writable directory.
=> merge config from ../satvision-toa-giant-patch8-window8-128/mim_pretrain_swinv2_satvision_giant_128_window08_50ep.yaml
Building un-initialized model
Successfully built uninitialized model
Attempting to load checkpoint from ../satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt
Successfully applied checkpoint
Calling model.encoder() for 128 samples to run prediction and calculate reconstruction losses
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [00:17<00:00,  7.27it/s]
Successfully exported reconstruction results to: ./image-reconstruction-example.pdf
```

