# pytorch-caney

Python package for lots of Pytorch tools.

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

## References

- [Pytorch Lightning](https://github.com/Lightning-AI/lightning)
