#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from multiprocessing import Manager
from multiprocessing import Pool
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
import torch

from liteasr import tasks
from liteasr.config import LiteasrConfig
from liteasr.tasks import LiteasrTask
from liteasr.train import config_init
from liteasr.utils.checkpoint import load_ckpt
from liteasr.utils.score import levenshtein

logger = logging.getLogger("liteasr.infer")


@hydra.main(config_path="config", config_name="config")
def main(cfg: LiteasrConfig) -> None:
    # make hydra logging work with ddp
    # (see https://github.com/facebookresearch/hydra/issues/1126)
    with open_dict(cfg):
        cfg.job_logging_cfg = OmegaConf.to_container(
            HydraConfig.get().job_logging, resolve=True
        )
    cfg = OmegaConf.create(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    )

    infer(cfg)


def dist_infer(
    rank: int,
    world_size: int,
    task: LiteasrTask,
    model: torch.nn.Module,
    dataset,
    tl: List[int],
    te: List[int],
):
    torch.set_num_threads(1)

    model.eval()
    with torch.no_grad():
        for data in dataset[rank::world_size]:
            feat = data.x.unsqueeze(0)
            ref = data.text
            hyp = task.inference(feat, model)
            res = "[X]" if ref == hyp else "[ ]"
            err = levenshtein(ref, hyp)
            tl[rank] += len(ref)
            te[rank] += err
            logger.info("\n{} {}\n{:>3} {}".format(res, hyp, err, ref))


def infer_dataset(
    task: LiteasrTask,
    model: torch.nn.Module,
    dataset,
    world_size: int,
):
    mgr = Manager()
    total_len = mgr.list([0] * world_size)
    total_err = mgr.list([0] * world_size)

    pool = Pool(world_size)
    for rank in range(world_size):
        pool.apply_async(
            dist_infer,
            args=(
                rank,
                world_size,
                task,
                model,
                dataset,
                total_len,
                total_err,
            ),
        )
    pool.close()
    pool.join()

    logger.info(
        "Error rate: {} / {} = {:.2%}".format(
            sum(total_err), sum(total_len),
            sum(total_err) / sum(total_len)
        )
    )


def infer(cfg: LiteasrConfig):
    # set task
    task = tasks.setup_task(cfg.task)
    logger.info("setting {} task...".format(task.__class__.__name__))

    # load test data
    logger.info("1. load data...")
    task.load_dataset("test", task.cfg.test)

    # build model
    model = task.build_model(cfg.model)

    # load checkpoints
    model_state_dict = load_ckpt(cfg.inference)
    model.load_state_dict(model_state_dict)

    # inference
    for test_set in task.dataset("test"):
        infer_dataset(
            task,
            model,
            dataset=test_set,
            world_size=cfg.inference.thread_num,
        )


def cli_main():
    config_init()
    main()


if __name__ == "__main__":
    cli_main()
