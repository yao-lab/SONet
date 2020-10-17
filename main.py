import logging
import os
import sys

from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpuid
from trainer import Trainer

logger = logging.getLogger("ODENet-Experiments")


def main(config):
    logger.info(' '.join(sys.argv))
    logger.info(f'***START TRAINING {config.model}***'.upper())
    try:
        trainer = Trainer(config)
        trainer.train_and_test()
    except Exception as e:
        logger.error(e, exc_info=True)
        raise


if __name__ == '__main__':
    main(config)
