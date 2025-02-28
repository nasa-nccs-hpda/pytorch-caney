# Requirements

* Container platform _or_ virtual environment (e.g., anaconda). 
* GPU support.

_CPU support is limited and the author does not provide any guarantee of usability._

# Provided

* Docker container, which can also be converted to a Singularity container.
* Virtual environment [specification](requirements/environment_gpu.yml) 

CPU support is limited and the author does not provide any guarantee of usability.

## Architecture

The container is built on top of NGC NVIDIA PYTORCH containers.

This application is powered by PyTorch and PyTorch Lighning AI/ML backends.

## Installation
SatVision-TOA can be installed and used via anaconda environments and containers.

## Example to Download the Container via Singularity

```bash
module load singularity
singularity build --sandbox pytorch-caney docker://nasanccs/pytorch-caney:latest

## Example to Install Anaconda Environment

``` bash
git clone git@github.com:nasa-nccs-hpda/pytorch-caney.git
cd pytorch-caney; conda env create -f requirements/environment_gpu.yml;
conda activate pytorch-caney
```

## Container Usage

As an example, you can shell into the container:

```bash
singularity shell --nv -B <mounts> /path/to/container/pytorch-caney
```
## Example to Download the Container via Singularity
