## USER GUIDE 

SatVision-TOA is composed of packaged Python applications called pipelines, which weave together 
configurable modules to perform workflows.  The runtime instructions for these pipelines below, 
typically involve running a python command-line application with a single configuration file.  
We anticipate that users will only initiate pipelines that leverage our _published, pre-trained_ model.  
For example, (1) image reconstruction and (2) 3D cloud retrieval.  

_For completeness, we have also provided the training pipeline (3) that was used to pre-train_ 
_the SatVision-TOA model.  Note that this pipeline requires advanced GPU and storage requirements._

**_NOTE: The Getting Started section must be completed [first.](requirements/README.md)_**

## <b> Running SatVision-TOA Pipelines </b>

### <b> Command-Line Interface (CLI) </b>

To run tasks with **SatVision-TOA**, use the following command:

```bash
$ python pytorch-caney/pytorch_caney/ptc_cli.py --config-path <Path to config file>
```

### <b> Common CLI Arguments </b>
| Command-line-argument | Description                                         |Required/Optional/Flag | Default  | Example                  |
| --------------------- |:----------------------------------------------------|:---------|:---------|:--------------------------------------|
| `-config-path`                  | Path to training config                                   | Required | N/A      |`--config-path pytorch-caney/configs/3dcloudtask_swinv2_satvision_gaint_test.yaml`         |
| `-h, --help`               | show this help message and exit                  | Optional | N/a      |`--help`, `-h` |


## Pipeline 1.  Image Reconstruction

See Section _4.1 Image Reconstruction_ (https://arxiv.org/pdf/2411.17000) for reconstruction performance measurements.
* [Run Image Reconstruction with Pretrained Model](./USER_GUIDE_IMAGE.md)

## Pipeline 2. 3D Cloud Retrieval

See _4.2 3D Cloud Retrieval Downstream Task_ (https://arxiv.org/pdf/2411.17000) for prediction details.
* [Run 3D Cloud Task with Pretrained Model](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128#-examples-)

## Pipeline 3. Model Training
See _3.1 Developing a remote sensing pre-training dataset with MODIS TOA_ (https://arxiv.org/pdf/2411.17000) for methodology.
* [Run SatVision-TOA Pretraining from Scratch](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128)
