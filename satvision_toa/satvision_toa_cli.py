import os
import logging
import argparse


from lightning.pytorch import Trainer

from satvision_toa.configs.config import _C, _update_config_from_file
from satvision_toa.utils import get_strategy, get_distributed_train_batches
from satvision_toa.pipelines import PIPELINES, get_available_pipelines
from satvision_toa.datamodules import DATAMODULES, get_available_datamodules


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(config, output_dir):

    logging.info('Training')

    # Get the proper pipeline
    available_pipelines = get_available_pipelines()
    logging.info("Available pipelines:", available_pipelines)
    pipeline = PIPELINES[config.PIPELINE]
    logging.info(f'Using {pipeline}')
    ptlPipeline = pipeline(config)

    # Resume from checkpoint
    if config.MODEL.RESUME:
        logging.info(
            f'Attempting to resume from checkpoint {config.MODEL.RESUME}')
        ptlPipeline = pipeline.load_from_checkpoint(config.MODEL.RESUME)

    # Determine training strategy
    strategy = get_strategy(config)

    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        devices=-1,
        strategy=strategy,
        precision=config.PRECISION,
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=config.PRINT_FREQ,
        default_root_dir=output_dir,
    )

    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(
            config, trainer)

    if config.DATA.DATAMODULE:
        available_datamodules = get_available_datamodules()
        logging.info(f"Available data modules: {available_datamodules}")
        datamoduleClass = DATAMODULES[config.DATAMODULE]
        datamodule = datamoduleClass(config)
        logging.info(f'Training using datamodule: {datamodule}')
        trainer.fit(model=ptlPipeline, datamodule=datamodule)

    else:
        logging.info(
            'Training without datamodule, assuming data is set' +
            f' in pipeline: {ptlPipeline}')
        trainer.fit(model=ptlPipeline)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-path',
        type=str,
        help='Path to pretrained model config'
    )

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    output_dir = os.path.join(
        config.OUTPUT, config.MODEL.NAME, config.TAG)
    logging.info(f'Output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(
        output_dir,
        f"{config.TAG}.config.json"
    )

    with open(path, "w") as f:
        f.write(config.dump())

    logging.info(f"Full config saved to {path}")
    logging.info(config.dump())

    main(config, output_dir)
