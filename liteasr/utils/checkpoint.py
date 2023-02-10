"""Checkpoint."""

import glob
import logging
import os
import re

import torch

from liteasr.config import InferenceConfig

logger = logging.getLogger(__name__)


def load_ckpt(cfg: InferenceConfig):
    if not cfg.model_avg:
        logger.info(f"loading checkpoint: {cfg.ckpt_path}/{cfg.ckpt_name}")
        model_state_dict = torch.load(
            f"{cfg.ckpt_path}/model.ep.{cfg.ckpt_name}.pt",
            map_location=(lambda storage, loc: storage),
        )
    else:

        def model_state_avg(ckpts):
            # sum
            model_state_dict = None
            for ckpt in ckpts:
                states = torch.load(
                    ckpt,
                    map_location=(lambda storage, loc: storage),
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

        ckpts = sorted(glob.glob(f"{cfg.ckpt_path}/*"), key=os.path.getmtime)
        pos = ckpts.index(f"{cfg.ckpt_path}/model.ep.{cfg.ckpt_name}.pt")
        assert pos - cfg.avg_num + 1 >= 0

        if cfg.avg_policy is None:
            pickup_ckpts = ckpts[pos - cfg.avg_num + 1 : pos + 1]
        else:
            loss = []
            with open(cfg.avg_policy, "r") as log:
                for line in log.readlines():
                    line = line.strip()
                    match = re.match(r".*valid loss: ([\d\.]+)", line)
                    if match:
                        loss.append(float(match.group(1)))

            ckpt_loss = sorted(
                zip(ckpts[: pos + 1], loss[: pos + 1]),
                key=lambda cl: cl[1],
            )[: cfg.avg_num]
            pickup_ckpts, _ = zip(*ckpt_loss)

        info = "\n\t".join(pickup_ckpts)
        logger.info(f"loading average checkpoint from:\n\t{info}")
        model_state_dict = model_state_avg(pickup_ckpts)

    return model_state_dict
