#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra

from omegaconf import OmegaConf
import torch

from liteasr import tasks
from liteasr.config import LiteasrConfig


@hydra.main(config_path='config', config_name='config')
def main(cfg: LiteasrConfig) -> None:
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model = model.cuda(6)
    optim = task.build_optimizer(model.parameters(), cfg.optimizer)
    task.load_data()

    import time
    stt = time.time()
    loss = torch.nn.L1Loss()
    for epoch in range(100):
        for i in range(len(task.dataset)):
            batch, _ = task.dataset[i]
            batch = batch.cuda(6)
            yp = model(batch)
            tgt = torch.zeros_like(yp)
            output = loss(yp, tgt)
            print(output.data)
            output.backward()
            optim.step()
            optim.zero_grad()
    end = time.time()
    print(end - stt)


if __name__ == '__main__':
    main()
