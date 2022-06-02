#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
from multiprocessing import Manager
from multiprocessing import Pool
import os
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import open_dict
import torch

from liteasr import tasks
from liteasr.config import InferenceConfig
from liteasr.config import LiteasrConfig
from liteasr.tasks import LiteasrTask
from liteasr.train import config_init

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


def load_ckpt(cfg: InferenceConfig):
    if not cfg.model_avg:
        logger.info(f"loading checkpoint: {cfg.ckpt_path}/{cfg.ckpt_name}")
        model_state_dict = torch.load(
            f"{cfg.ckpt_path}/{cfg.ckpt_name}",
            map_location=lambda storage,
            loc: storage,
        )
    else:
        ckpts = sorted(glob.glob(f"{cfg.ckpt_path}/*"), key=os.path.getmtime)
        pos = ckpts.index(f"{cfg.ckpt_path}/{cfg.ckpt_name}")
        assert pos - cfg.avg_num + 1 >= 0
        pickup_ckpts = ckpts[pos - cfg.avg_num + 1:pos + 1]
        logger.info(f"loading average checkpoint from: {pickup_ckpts}")

        # sum
        model_state_dict = None
        for ckpt in pickup_ckpts:
            states = torch.load(
                ckpt,
                map_location=lambda storage,
                loc: storage,
            )
            if model_state_dict is None:
                model_state_dict = states
            else:
                for k in model_state_dict.keys():
                    model_state_dict[k] += states[k]

        # average
        for k in model_state_dict.keys():
            if model_state_dict[k] is not None:
                if model_state_dict[k].is_floating_point():
                    model_state_dict[k] /= cfg.avg_num
                else:
                    model_state_dict[k] //= cfg.avg_num

    return model_state_dict


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""

    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    curr = list(range(n + 1))
    for i in range(1, m + 1):
        prev, curr = curr, [i] + [0] * n
        for j in range(1, n + 1):
            insert, delete = prev[j] + 1, curr[j - 1] + 1
            change = prev[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            curr[j] = min(insert, delete, change)

    return curr[n]


def dist_infer(
    rank: int,
    world_size: int,
    task: LiteasrTask,
    model: torch.nn.Module,
    tl: List[int],
    te: List[int],
):
    torch.set_num_threads(1)

    model.eval()
    with torch.no_grad():
        for test_set in task.dataset("test"):
            for data in test_set[rank::world_size]:
                feat = data.x.unsqueeze(0)
                ref = data.text
                hyp = task.inference(feat, model)
                res = "[X]" if ref == hyp else "[ ]"
                err = levenshtein(ref, hyp)
                tl[rank] += len(ref)
                te[rank] += err
                logger.info("\n{} {}\n{:>3} {}".format(res, hyp, err, ref))


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

    # multi-thread inference
    mgr = Manager()
    total_len = mgr.list([0] * cfg.inference.thread_num)
    total_err = mgr.list([0] * cfg.inference.thread_num)

    pool = Pool(cfg.inference.thread_num)
    for rank in range(cfg.inference.thread_num):
        pool.apply_async(
            dist_infer,
            args=(
                rank,
                cfg.inference.thread_num,
                task,
                model,
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


def cli_main():
    config_init()
    main()


if __name__ == "__main__":
    cli_main()
