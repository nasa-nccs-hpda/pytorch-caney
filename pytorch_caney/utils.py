from lightning.pytorch.strategies import DeepSpeedStrategy


# -----------------------------------------------------------------------------
# get_strategy
# -----------------------------------------------------------------------------
def get_strategy(config):

    strategy = config.TRAIN.STRATEGY

    if strategy == 'deepspeed':
        deepspeed_config = {
            "train_micro_batch_size_per_gpu": config.DATA.BATCH_SIZE,
            "steps_per_print": config.PRINT_FREQ,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": config.DEEPSPEED.STAGE,
                "contiguous_gradients":
                    config.DEEPSPEED.CONTIGUOUS_GRADIENTS,
                "overlap_comm": config.DEEPSPEED.OVERLAP_COMM,
                "reduce_bucket_size": config.DEEPSPEED.REDUCE_BUCKET_SIZE,
                "allgather_bucket_size":
                    config.DEEPSPEED.ALLGATHER_BUCKET_SIZE,
            },
            "activation_checkpointing": {
                "partition_activations": config.TRAIN.USE_CHECKPOINT,
            },
        }

        return DeepSpeedStrategy(config=deepspeed_config)

    else:
        # These may be return as strings
        return strategy


# -----------------------------------------------------------------------------
# get_distributed_train_batches
# -----------------------------------------------------------------------------
def get_distributed_train_batches(config, trainer):
    if config.TRAIN.NUM_TRAIN_BATCHES:
        return config.TRAIN.NUM_TRAIN_BATCHES
    else:
        return config.DATA.LENGTH // \
            (config.DATA.BATCH_SIZE * trainer.world_size)
