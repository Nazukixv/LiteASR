"""Trainer."""

import logging
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from liteasr.config import LiteasrConfig
from liteasr.criterions import LiteasrLoss
from liteasr.distributed.ddp_model_wrapper import DDPModelWrapper
from liteasr.models import LiteasrModel
from liteasr.optims import LiteasrOptimizer
from liteasr.tasks import LiteasrTask
from liteasr.utils.data_loader import EpochDataLoader
from liteasr.utils.device import to_device
from liteasr.utils.trigger import EventManager
from liteasr.utils.trigger import Trigger

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(
        self,
        cfg: LiteasrConfig,
        task: LiteasrTask,
        model: LiteasrModel,
        criterion: LiteasrLoss,
        optimizer: LiteasrOptimizer,
    ):
        self.cfg = cfg
        self.task = task
        self._model = model
        self._wrapped_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.iter = 0

        if self.cfg.distributed.world_size > 1:
            train_sampler = DistributedSampler(self.task.dataset("train"))
            valid_sampler = DistributedSampler(self.task.dataset("valid"))
        else:
            train_sampler = None
            valid_sampler = None

        self.train_iter = EpochDataLoader(
            dataset=self.task.dataset("train"),
            batch_size=1,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=lambda x: x[0],
        )
        self.valid_iter = DataLoader(
            dataset=self.task.dataset("valid"),
            batch_size=1,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=lambda x: x[0],
        )

        self.device = torch.device("cuda")

        self.event_manager = EventManager()
        self.event_manager.add_event(self._report_loss)
        self.event_manager.add_event(self._valid)
        self.event_manager.add_event(self._save_model)
        self.loss = None

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.cfg.distributed.world_size > 1:
                ddp_model = DistributedDataParallel(
                    module=self._model,
                    device_ids=[self.cfg.distributed.device_id],
                    output_device=self.cfg.distributed.rank,
                )
                self._wrapped_model = DDPModelWrapper(ddp_model)
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def epoch(self):
        return self.train_iter.epoch

    @property
    def max_epoch(self):
        if self.cfg.optimization.max_epoch > 0:
            return self.cfg.optimization.max_epoch
        else:
            return "inf"

    @property
    def max_iter(self):
        if self.cfg.optimization.max_iter > 0:
            return self.cfg.optimization.max_iter
        else:
            return "inf"

    def is_master(self):
        if self.cfg.distributed.world_size > 1:
            return dist.get_rank() == 0
        else:
            return True

    def stop(self):
        reach_max_epoch = (
            self.cfg.optimization.max_epoch >= 0
            and self.epoch >= self.cfg.optimization.max_epoch
        )
        reach_max_iter = (
            self.cfg.optimization.max_iter >= 0
            and self.iter >= self.cfg.optimization.max_iter
        )
        return reach_max_epoch or reach_max_iter

    def run(self):
        for batch in self.train_iter:
            # trigger epoch-wise events
            self.event_manager.trigger_events("epoch")

            # stop training process if reach limit
            if self.stop():
                break

            # train one step
            batch = to_device(batch, self.device)

            loss = self.criterion(self.model, *batch)
            self.loss = loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5.0
            )
            if not math.isnan(grad_norm):
                self.optimizer.step()
                self.iter += 1
            self.optimizer.zero_grad()

            # trigger iteration-wise events
            self.event_manager.trigger_events("iteration")

    @Trigger(100, "iteration")
    def _report_loss(self):
        logger.info(
            "{} / {} iters, {} / {} epochs - current loss: {:.2f}".format(
                self.iter,
                self.max_iter,
                self.epoch,
                self.max_epoch,
                self.loss.item(),
            )
        )

    @Trigger(1, "epoch")
    def _valid(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for bat in self.valid_iter:
                bat = to_device(bat, self.device)
                loss = self.criterion(self.model, *bat)
                if self.cfg.distributed.world_size > 1:
                    dist.reduce(loss, dst=0)
                    loss /= self.cfg.distributed.world_size
                losses.append(loss)
            reduced_loss = torch.mean(torch.tensor(losses))
            logger.info(
                "{} / {} iters, {} / {} epochs - valid loss: {:.2f}".format(
                    self.iter,
                    self.max_iter,
                    self.epoch,
                    self.max_epoch,
                    reduced_loss.item(),
                )
            )
        self.model.train()

    @Trigger(1, "epoch")
    def _save_model(self):
        if self.is_master():
            model_name = "model.ep.{}.pt".format(self.epoch)
            self.task.save_model(model_name, self.model)
