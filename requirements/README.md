# Getting Started

## Requirements

* Container platform _or_ virtual environment (e.g., anaconda). 
* GPU support.
* HuggingFace account

_CPU support is limited and the author does not provide any guarantee of usability._

## Provided

* Docker container, which can also be converted to a Singularity container.
* Virtual environment specification [file](environment_gpu.yml) 

CPU support is limited and the author does not provide any guarantee of usability.

## Architecture

The container is built on top of NGC NVIDIA PYTORCH containers.

This application is powered by PyTorch and PyTorch Lighning AI/ML backends.

## Installation

SatVision-TOA can be installed in at least two ways: 1) Singularity container or 2) Anaconda environment.

### 1) Singularity Container Installation

```bash
module load singularity
singularity build --sandbox pytorch-caney docker://nasanccs/pytorch-caney:latest
```

#### Container Usage

As an example, you can shell into the container:

```bash
singularity shell --nv -B <mounts> /path/to/container/pytorch-caney
Singularity> python <SatVision-TOA API>
```

### 2) Anaconda Environment Installation

``` bash
git clone git@github.com:nasa-nccs-hpda/pytorch-caney.git
cd pytorch-caney; conda env create -f requirements/environment_gpu.yml;
```

#### Environment Usage

```bash
conda activate pytorch-caney
python <SatVision-TOA API>
```

## SatVision-TOA Application Programming Interface (API) 

The API for SatVision-TOA, which is a Python application that is invoked from the command line, is described in the [User Guide](../USER_GUIDE.md).

_Note that the only runtime difference based on installation is that the command line is prefixed by **Singularity>** when invoked from within the container_


TBD
- Add hugging face certificate stuff here....
- We are missing a file from the HF repo:  