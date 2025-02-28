## USER GUIDE 

SatVision-TOA is composed of packaged Python applications called pipelines, which weave together 
configurable modules to perform workflows.  The runtime instructions for these pipelines below, 
typically involve running a python command-line application with a single configuration file.  
This configuration file specifies all workflow parameters and control conditions.

We anticipate that users will only initiate pipelines that leverage our _published, pre-trained model_.  
For example (1) image reconstruction and (2) 3D cloud retrieval.  

For completeness, we have also provided the training workflow (3) that was used to pre-train 
the SatVision-TOA model.  Note that this pipeline requires advanced GPU and storage requirements.

_NOTE:  Verify [installation instructions](requirements/README.md) before proceeding._

## 1.  Image Reconstruction
* [Run Image Reconstruction with Pretrained Model](TBD)

## 3D Cloud Retrieval

* [Run 3D Cloud Task with Pretrained Model](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128#-examples-)
* [Run 3D Cloud Task with baseline model](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128#-examples-)

## Model Training

* [Run SatVision-TOA Pretraining from Scratch](https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128)
