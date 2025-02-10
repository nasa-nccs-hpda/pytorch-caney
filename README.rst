================
pytorch-caney
================

Python package for lots of Pytorch tools for geospatial science problems.

.. image:: https://zenodo.org/badge/472450059.svg
      :target: https://zenodo.org/badge/latestdoi/472450059

Objectives
------------

- Library to process remote sensing imagery using GPU and CPU parallelization.
- Machine Learning and Deep Learning image classification and regression.
- Agnostic array and vector-like data structures.
- User interface environments via Notebooks for easy to use AI/ML projects.
- Example notebooks for quick AI/ML start with your own data.

Installation
----------------

The following library is intended to be used to accelerate the development of data science products
for remote sensing satellite imagery, or any other applications. pytorch-caney can be installed
by itself, but instructions for installing the full environments are listed under the requirements
directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure NVIDIA libraries
are installed locally in the system if not using conda/mamba.

.. code-block:: bash

    module load singularity # if a module needs to be loaded
    singularity build --sandbox pytorch-caney-container docker://nasanccs/pytorch-caney:latest


Why Caney?
---------------

"Caney" means longhouse in Taíno.

Contributors
-------------

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Jian Li, jian.li@nasa.gov
- Caleb Spradlin, cspradlindev@gmail.com

Contributing
-------------

Please see our `guide for contributing to pytorch-caney <CONTRIBUTING.md>`_.

SatVision
------------

+---------------+--------------+------------+------------+
| Name          | Pretrain     | Resolution | Parameters |
+===============+==============+============+============+
| SatVision-B   | MODIS-1.9-M  | 192x192    | 84.5M      |
+---------------+--------------+------------+------------+

SatVision Datasets
-----------------------

+---------------+-----------+------------+-------------+
| Name          | Bands     | Resolution | Image Chips |
+===============+===========+============+=============+
| MODIS-Small   | 7         | 128x128    | 1,994,131   |
+---------------+-----------+------------+-------------+

MODIS Surface Reflectance (MOD09GA) Band Details
------------------------------------------------------

+-----------------+---------------+
| Band Name       | Bandwidth     |
+=================+===============+
| sur_refl_b01_1  | 0.620 - 0.670 |
+-----------------+---------------+
| sur_refl_b02_1  | 0.841 - 0.876 |
+-----------------+---------------+
| sur_refl_b03_1  | 0.459 - 0.479 |
+-----------------+---------------+
| sur_refl_b04_1  | 0.545 - 0.565 |
+-----------------+---------------+
| sur_refl_b05_1  | 1.230 - 1.250 |
+-----------------+---------------+
| sur_refl_b06_1  | 1.628 - 1.652 |
+-----------------+---------------+
| sur_refl_b07_1  | 2.105 - 2.155 |
+-----------------+---------------+

Pre-training with Masked Image Modeling
-----------------------------------------

To pre-train the swinv2 base model with masked image modeling pre-training, run:

.. code-block:: bash

    torchrun --nproc_per_node <NGPUS> pytorch-caney/pytorch_caney/pipelines/pretraining/mim.py --cfg <config-file> --dataset <dataset-name> --data-paths <path-to-data-subfolder-1> --batch-size <batch-size> --output <output-dir> --enable-amp

For example to run on a compute node with 4 GPUs and a batch size of 128 on the MODIS SatVision pre-training dataset with a base swinv2 model, run:

.. code-block:: bash

    singularity shell --nv -B <mounts> /path/to/container/pytorch-caney-container
    Singularity> export PYTHONPATH=$PWD:$PWD/pytorch-caney
    Singularity> torchrun --nproc_per_node 4 pytorch-caney/pytorch_caney/pipelines/pretraining/mim.py --cfg pytorch-caney/examples/satvision/mim_pretrain_swinv2_satvision_base_192_window12_800ep.yaml --dataset MODIS --data-paths /explore/nobackup/projects/ilab/data/satvision/pretraining/training_* --batch-size 128 --output . --enable-amp


This example script runs the exact configuration used to make the SatVision-base model pre-training with MiM and the MODIS pre-training dataset.

.. code-block:: bash

    singularity shell --nv -B <mounts> /path/to/container/pytorch-caney-container
    Singularity> cd pytorch-caney/examples/satvision
    Singularity> ./run_satvision_pretrain.sh


Fine-tuning Satvision-base
-----------------------------

To fine-tune the satvision-base pre-trained model, run:

.. code-block:: bash

    torchrun --nproc_per_node <NGPUS> pytorch-caney/pytorch_caney/pipelines/finetuning/finetune.py --cfg <config-file> --pretrained <path-to-pretrained> --dataset <dataset-name> --data-paths <path-to-data-subfolder-1> --batch-size <batch-size> --output <output-dir> --enable-amp

See example config files pytorch-caney/examples/satvision/finetune_satvision_base_*.yaml to see how to structure your config file for fine-tuning.


Testing
------------

For unittests, run this bash command to run linting and unit test runs. This will execute unit tests and linting in a temporary venv environment only used for testing.

.. code-block:: bash

    git clone git@github.com:nasa-nccs-hpda/pytorch-caney.git
    cd pytorch-caney; bash test.sh


or run unit tests directly with container or anaconda env

.. code-block:: bash

    git clone git@github.com:nasa-nccs-hpda/pytorch-caney.git
    singularity build --sandbox pytorch-caney-container docker://nasanccs/pytorch-caney:latest
    singularity shell --nv -B <mounts> /path/to/container/pytorch-caney-container
    cd pytorch-caney; python -m unittest discover pytorch_caney/tests

.. code-block:: bash

    git clone git@github.com:nasa-nccs-hpda/pytorch-caney.git
    cd pytorch-caney; conda env create -f requirements/environment_gpu.yml;
    conda activate pytorch-caney
    python -m unittest discover pytorch_caney/tests


References
------------

- `Pytorch Lightning <https://github.com/Lightning-AI/lightning>`_ 
- `Swin Transformer <https://github.com/microsoft/Swin-Transformer>`_ 
- `SimMIM <https://github.com/microsoft/SimMIM>`_ 
