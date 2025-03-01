## USER GUIDE 

SatVision-TOA is composed of packaged Python applications called pipelines, which weave together 
configurable modules to perform workflows.  The runtime instructions for these pipelines below, 
typically involve running a python command-line application with a single configuration file.  

We anticipate that users will only initiate pipelines that leverage our _published, pre-trained_ model.  For example, (1) image reconstruction and (2) 3D cloud retrieval.  

_For completeness, we have also provided the training pipeline (3) that was used to pre-train_ 
_the SatVision-TOA model.  Note that this pipeline requires advanced GPU and storage requirements._

### PIPELINES 

**_NOTE:  Complete [installation instructions](requirements/README.md) before proceeding._**

## Pipeline 1.  Image Reconstruction

See Section _4.1 Image Reconstruction_ (https://arxiv.org/pdf/2411.17000) for reconstruction performance measurements.
* [Run Image Reconstruction with Pretrained Model](TBD)

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

## Pipeline 2. 3D Cloud Retrieval

See _4.2 3D Cloud Retrieval Downstream Task_ (https://arxiv.org/pdf/2411.17000) for prediction details.
* [Run 3D Cloud Task with Pretrained Model](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128#-examples-)
* [Run 3D Cloud Task with baseline model](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128#-examples-)

## Pipeline 3. Model Training
See _3.1 Developing a remote sensing pre-training dataset with MODIS TOA_ for methodology.
* [Run SatVision-TOA Pretraining from Scratch](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128)
