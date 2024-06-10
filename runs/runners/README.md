# Runners

softlink these to cwd with pytorch-caney

## Discover

For starting a model from scratch
```bash
$ sbatch discover_svtoa_pretraining_runner.sh mim_pretrain_swinv2_satvision_huge_128_window8_patch8_onecycle_100ep.yaml
```

For starting resuming a model

```bash
$ sbatch discover_svtoa_pretraining_runner_resume.sh mim_pretrain_swinv2_satvision_huge_128_window8_patch8_onecycle_100ep.yaml mim_satvision_pretrain-huge/mim_pretrain_swinv2_h_satvision_128_window8_mpatch8_100ep/ckpt_epoch_60
```

- !! Deepspeed does not allow resuming training with a different world size (n gpus). You must use the same number of gpus (/nodes) when resuming.
- The 2nd arg should be the path to the checkpoint directory without a trailing `/`. I haven't edited to code to handle this obvious bug.


## Frontier

For starting model from scratch
- same as above commands just switch out the discover runner script for the frontier runner script.
- If needed, switch out the cast command to copy from here: /lustre/orion/geo160/proj-shared/envs/rocm-torch-test-full-0.1.0.tar.gz instead of where it currently is.

## !! Note
For 26m and 100m dataset, make sure to increase NUM_SAMPLES in the mim_deepspeed.py script, that reflects dataset length.
