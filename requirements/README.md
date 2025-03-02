# Getting Started

In addition to the instructions below, GPU support is required to effectively run SatVision-TOA.  

_NOTE: CPU support is limited and the author does not provide any guarantee of usability._

## Provided

* Docker container, which can also be converted to a Singularity container.
* Virtual environment specification [file](environment_gpu.yml) 

## Architecture

* The container is built on top of NGC NVIDIA PYTORCH containers.
* This application is powered by PyTorch and PyTorch Lighning AI/ML backends.

## Instructions

1. Install container platform _or_ virtual environment (e.g., anaconda). 
2. Create a Hugging Face account.
3. Download SatVision-TOA datasets.

### 1. Install container platform _or_ virtual environment (e.g., anaconda). 

SatVision-TOA can be installed in at least two ways: 1) Singularity container or 2) Anaconda environment.

#### 1(a) Singularity Container Installation

```bash
module load singularity
singularity build --sandbox pytorch-caney docker://nasanccs/pytorch-caney:latest
```

##### Container Usage

As an example, you can shell into the container:

```bash
singularity shell --nv -B <mounts> /path/to/container/pytorch-caney
Singularity> python <SatVision-TOA API>
```

#### 1(b) Anaconda Environment Installation

``` bash
git clone --single-branch --branch docs https://github.com/nasa-nccs-hpda/pytorch-caney.git
cd pytorch-caney; conda env create -f requirements/environment_gpu.yml;
```

##### Environment Usage

```bash
conda activate pytorch-caney
python <SatVision-TOA API>
```

##### SatVision-TOA Application Programming Interface (API) 

The API for SatVision-TOA, which is a Python application that is invoked from the command line, is described in the [User Guide](../USER_GUIDE.md).

_Note that the only runtime difference based on installation is that the command line is prefixed by **Singularity>** when invoked from within the container._

### 2. Create a Hugging Face account

A Hugging Face account is required in order to retrieve the model and supporting datasets.  Required steps:
a. Create HF account.
b. Create a local SSH key.
c. Register the SSH key with HF.

#### 2(a) Create HF account
Visit the [website](https://huggingface.co/join) to create HF account by specifying the "_<e-mail address>_" to link to account.

#### 2(b) Create a local SSH key 
Perform the shortcut steps in the sample session below to create an SSH key and add it to HF.  Background details are provided here: https://huggingface.co/docs/hub/en/security-git-ssh#add-a-ssh-key-to-your-account

### _Sample Session - Create SSH key and add to Hugging Face_ 

```bash
(base) gtamkin@gpu004:/explore/nobackup/projects/ilab/projects/Satvision
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ ssh-keygen -t ed25519 -C "_<e-mail address>_"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/<user>/.ssh/id_ed25519): /home/<user>/.ssh/id_satvision-toa-test
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 

Your identification has been saved in /home/<user>/.ssh/id_satvision-toa-test.
Your public key has been saved in /home/<user>.ssh/id_satvision-toa-test.pub.
The key fingerprint is:
SHA256:2ztflrt/OxUFW4UKlOBQQzrnpoVPTLbEo3wZJ/XXXXXX _<e-mail address>_ 
The key's randomart image is:
+--[ED25519 256]--+
|      .o=o..  .o+|
|       = .o   .o.|
|      o @ .. .. .|
|     . & B  .  . |
|      E S       .|
|       O +     ..|
+----[SHA256]-----+

<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ ls -alt ~/.ssh/id_satvision-*
-rw------- 1 <user> ilab 105 Mar  2 08:20 /home/<user>/.ssh/id_satvision-toa-test.pub
-rw------- 1 <user> ilab 419 Mar  2 08:20 /home/<user>/.ssh/id_satvision-toa-test

<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ more /home/<user>/.ssh/id_satvision-toa-test.pub
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHt6HC5R4gT2ZwUg8zqijhNj4Op86isIfY2LXXXXX _<e-mail address>_
```

#### 2(c) Register the SSH key with HF 

Cut and paste the contents of the .pub file into the GUI: https://huggingface.co/settings/keys/add?type=ssh, which looks like this:

![https://huggingface.co/settings/keys/add?type=ssh](ssh.png)

if you encounter problems, consult: https://huggingface.co/docs/hub/en/security-git-ssh#add-a-ssh-key-to-your-account

### 3. Download SatVision-TOA datasets

After completing steps #1 and #2 above, acquire the pre-requisite datasets.  Although certain artifacts are only needed for certain tasks, 
we download the superset of dependencies now for simplicity.  By default, SatVision-TOA uses relative directories to resolve input data paths
at runtime.  So, we suggest that all installation steps occur from the same root directory, as the sample session suggests.

* Model Repository: https://huggingface.co/nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
* Dataset repo: https://huggingface.co/datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation

### _Sample Session - Download SatVision-TOA datasets_ 

```bash
<user>@gpu004:~$ cd /explore/nobackup/projects/ilab/projects/Satvision
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone --single-branch --branch docs https://github.com/nasa-nccs-hpda/pytorch-caney.git
Cloning into 'pytorch-caney'...
remote: Enumerating objects: 1337, done.
remote: Counting objects: 100% (176/176), done.
remote: Compressing objects: 100% (103/103), done.
remote: Total 1337 (delta 104), reused 90 (delta 73), pack-reused 1161 (from 1)
Receiving objects: 100% (1337/1337), 20.08 MiB | 15.85 MiB/s, done.
Resolving deltas: 100% (749/749), done.
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ module load git-lfs
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git lfs install
Updated Git hooks.
Git LFS initialized.
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
Cloning into 'satvision-toa-giant-patch8-window8-128'...
remote: Enumerating objects: 28, done.
remote: Counting objects: 100% (24/24), done.
remote: Compressing objects: 100% (24/24), done.
remote: Total 28 (delta 12), reused 0 (delta 0), pack-reused 4 (from 1)
Receiving objects: 100% (28/28), 14.75 KiB | 14.75 MiB/s, done.
Resolving deltas: 100% (12/12), done.
Encountered 1 file that may not have been copied correctly on Windows:
	mp_rank_00_model_states.pt

See: `git lfs help smudge` for more details.
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation
Cloning into 'modis_toa_cloud_reconstruction_validation'...
remote: 
remote: ========================================================================
remote: 
remote: ERROR: Repository not found

remote: 
remote: ========================================================================
remote: 
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ 
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone --single-branch --branch docs https://github.com/nasa-nccs-hpda/pytorch-caney.git
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ module load git-lfs
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git lfs install
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128
<user>@gpu004:/explore/nobackup/projects/ilab/projects/Satvision$ git clone git@hf.co:datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation

```

NOTES: 
- After vetting, the docs branch will be migrated into the main branch.  At this point, we can drop the ```--single-branch --branch docs``` parameter
- We are missing this file from the HF repo (Jordan?):  datasets/nasa-cisto-data-science-group/modis_toa_cloud_reconstruction_validation.  Image reconstruction cannot be run until this is restored.