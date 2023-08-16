# pytorch-caney

Python package for lots of Pytorch tools.

[![DOI](https://zenodo.org/badge/472450059.svg)](https://zenodo.org/badge/latestdoi/472450059)
![CI Workflow](https://github.com/nasa-nccs-hpda/pytorch-caney/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub ](https://github.com/nasa-nccs-hpda/pytorch-caney/actions/workflows/dockerhub.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/pytorch-caney/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/pytorch-caney/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/pytorch-caney?branch=main)

## Objectives

- Library to process remote sensing imagery using GPU and CPU parallelization.
- Machine Learning and Deep Learning image classification and regression.
- Agnostic array and vector-like data structures.
- User interface environments via Notebooks for easy to use AI/ML projects.
- Example notebooks for quick AI/ML start with your own data.

## Installation

The following library is intended to be used to accelerate the development of data science products
for remote sensing satellite imagery, or any other applications. pytorch-caney can be installed
by itself, but instructions for installing the full environments are listed under the requirements
directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure NVIDIA libraries
are installed locally in the system if not using conda/mamba.

```bash
module load singularity
singularity build --sandbox pytorch-caney docker://nasanccs/pytorch-caney:latest
```

## Why Caney?

"Caney" means longhouse in Taíno.

## Contributors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Caleb Spradlin, caleb.s.spradlin@nasa.gov

## Contributing

Please see our [guide for contributing to pytorch-caney](CONTRIBUTING.md).

## SatVision

| name | pretrain | resolution | #params |
| :---: | :---: | :---: | :---: |
| SatVision-B | MODIS-1.9-M | 192x192 | 84.5M |

## SatVision Datasets

| name | bands | resolution | #params |
| :---: | :---: | :---: | :---: |
| MODIS-Small | 7 | 128x128 | 1,994,131 |

## Pre-training with Masked Image Modeling
To pre-train the swinv2 base model with masked image modeling pre-training, run:
```bash
torchrun --nproc_per_node <NGPUS> --cfg <config-file> --dataset <dataset-name> --data-paths <path-to-data-subfolder-1> --batch-size <batch-size> --output <output-dir> --enable-amp
```

For example to run on a compute node with 4 GPUs and a batch size of 128 on the MODIS SatVision pre-training dataset with a base swinv2 model, run:

```bash
singularity shell --nv -B <mounts> /path/to/container/pytorch-caney
Singularity> export PYTHONPATH=$PWD:$PWD/pytorch-caney
Singularity> torchrun --nproc_per_node 4 pytorch-caney/pytorch_caney/pipelines/pretraining/mim.py --cfg pytorch-caney/examples/satvision/mim_pretrain_swinv2_satvision_base_192_window12_800ep.yaml --dataset MODIS --data-paths /explore/nobackup/projects/ilab/data/satvision/pretraining/training_* --batch-size 128 --output . --enable-amp
```

This example script runs the exact configuration used to make the SatVision-base model pre-training with MiM and the MODIS pre-training dataset.
```bash
singularity shell --nv -B <mounts> /path/to/container/pytorch-caney
Singularity> cd pytorch-caney/examples/satvision
Singularity> ./run_satvision_pretrain.sh
```

## References

- [Pytorch Lightning](https://github.com/Lightning-AI/lightning)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [SimMIM](https://github.com/microsoft/SimMIM)
