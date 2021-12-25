#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra
import torch

from liteasr import tasks
from liteasr.config import LiteasrConfig
from liteasr.criterions import LiteasrLoss
from liteasr.models import LiteasrModel
from liteasr.optims import LiteasrOptimizer
from liteasr.tasks import LiteasrTask


@hydra.main(config_path="config", config_name="config")
def main(cfg: LiteasrConfig) -> None:
    # set random seed
    torch.manual_seed(42)

    # set torch device
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    # set task
    task = tasks.setup_task(cfg.task)

    # load training data
    task.load_data()

    # build model
    model = task.build_model(cfg.model)
    model = model.to(device=device)

    # build optimizer
    optim = task.build_optimizer(model.parameters(), cfg.optimizer)

    # build criterion
    criter = task.build_criterion(cfg.criterion)

    train(task, model, optim, criter, device=device, epoch=-1)


def train(
    task: LiteasrTask,
    model: LiteasrModel,
    optim: LiteasrOptimizer,
    criter: LiteasrLoss,
    device: torch.device,
    epoch: int,
):
    # epoch < 0: infinite loop
    # epoch = 0: do nothing
    # epoch > 0: train for `epoch` epoches
    ep = 0
    while epoch < 0 or ep < epoch:
        # train
        for _, batch in enumerate(task.dataset):
            xs_pad, xlens, ys_pad, ylens = batch
            xs_pad = xs_pad.to(device=device)
            ys_pad = ys_pad.to(device=device)

            loss = criter(model, xs_pad, xlens, ys_pad, ylens)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            optim.zero_grad()

        # evaluate
        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                print("\033[%d;%dH" % (1, 0))
                i = 0
                for data_batch in task.data:
                    for test_data in data_batch:
                        feats = test_data.x.unsqueeze(0).to(device=device)
                        tgt = "".join(task.vocab.lookup(test_data.tokenids))
                        pred = task.inference(feats, model=model)
                        print(" " * 80)
                        print("\033[%d;%dH" % (i + 1, 0))
                        print(f"{'✅' if tgt == pred else '💔'} {pred}")
                        i += 1
                print("=" * 80)
                print(
                    "{} LOSS: {:>9.4f} GRAD: {:9.4f}".format(
                        ep, loss.item(), grad_norm.item()
                    )
                )
            model.train()

        ep += 1


if __name__ == "__main__":
    main()
