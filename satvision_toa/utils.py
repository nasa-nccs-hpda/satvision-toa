import argparse
from datetime import datetime
from huggingface_hub import hf_hub_download
from lightning.pytorch.strategies import DeepSpeedStrategy
from satvision_toa.configs.config import _C, _update_config_from_file


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


# -------------------------------------------------------------------------
# validate_date
# -------------------------------------------------------------------------
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


# -------------------------------------------------------------------------
# load_config
# -------------------------------------------------------------------------
def load_config(
            model_repo_id: str = ('nasa-cisto-data-science-group/' 
                           'satvision-toa-giant-patch8-window8-128'),
            config_filename: str = ('mim_pretrain_swinv2_satvision_giant'
                             '_128_window08_50ep.yaml'),
            model_filename: str = 'mp_rank_00_model_states.pt'
        ):
    """
        Loads the mim-model config for SatVision from HF.

        Returns:
            config: reference to config file that can be used to load the model
    """

    # Extract filenames from HF to be used later
    config_filename = hf_hub_download(
        repo_id=model_repo_id,
        filename=config_filename)
    ckpt_model_filename = hf_hub_download(  # CHANGE
        repo_id=model_repo_id, 
        filename=model_filename)

    # edit config so we can load mim model from it
    config = _C.clone()
    _update_config_from_file(config, config_filename)
    config.defrost()
    config.MODEL.RESUME = ckpt_model_filename
    config.freeze()

    return config
