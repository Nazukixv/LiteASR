#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
import torch

from liteasr import tasks
from liteasr.config import LiteasrConfig
from liteasr.distributed import utils as dist_util
from liteasr.trainer import Trainer

logger = logging.getLogger("liteasr.train")


@hydra.main(config_path="config", config_name="config")
def main(cfg: LiteasrConfig) -> None:
    # make hydra logging work with ddp
    # (see https://github.com/facebookresearch/hydra/issues/1126)
    with open_dict(cfg):
        cfg.job_logging_cfg = OmegaConf.to_container(
            HydraConfig.get().job_logging, resolve=True
        )
        cfg.run_cfg = OmegaConf.to_container(
            HydraConfig.get().run, resolve=True
        )
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    )
    # OmegaConf.set_struct(cfg, True)

    dist_util.call_func(train, cfg)


def config_init(config_name="liteasr_config"):
    cs = ConfigStore.instance()
    cs.store(name=config_name, node=LiteasrConfig)


def cli_main():
    config_init()
    main()


def train(cfg: LiteasrConfig):
    # set random seed
    torch.manual_seed(cfg.common.seed)
    logger.info("set random seed as {}".format(cfg.common.seed))

    # set torch device
    device = torch.device("cuda")

    # set task
    task = tasks.setup_task(cfg.task)
    logger.info("setting {} task...".format(task.__class__.__name__))

    # load training data
    logger.info("1. load data...")
    if not cfg.common.memory_save:
        task.load_dataset("train", task.cfg.train, cfg.dataset, cfg.postprocess)
        task.load_dataset("valid", task.cfg.valid, cfg.dataset, cfg.postprocess)
    else:
        assert dist_util.get_world_size() > 1

        logger.info("1.1 split data...")
        dist_util.switch_logger_level()
        for rank in range(dist_util.get_world_size()):
            if dist_util.get_rank() == rank:
                task.load_dataset(
                    "train",
                    task.cfg.train,
                    cfg.dataset,
                    cfg.postprocess,
                    memory_save=cfg.common.memory_save,
                )
            dist_util.barrier()
        dist_util.switch_logger_level()
        task.load_dataset("valid", task.cfg.valid, cfg.dataset, cfg.postprocess)

    # build model
    model = task.build_model(cfg.model)
    model = model.to(device=device)
    logger.info("2. build model    : {}".format(model.__class__.__name__))
    logger.debug("model structure:\n{}".format(model))

    # build optimizer
    optim = task.build_optimizer(model.parameters(), cfg.optimizer)
    logger.info("3. build optimizer: {}".format(optim.__class__.__name__))

    # build criterion
    criter = task.build_criterion(cfg.criterion)
    logger.info("4. build criterion: {}".format(criter.__class__.__name__))

    # show final config
    repr_cfg = "config:\n  " + OmegaConf.to_yaml(cfg).replace("\n", "\n  ")
    logger.debug("model training config:\n{}".format(repr_cfg))

    trainer = Trainer(cfg, task, model, criter, optim)
    trainer.run()


if __name__ == "__main__":
    cli_main()
