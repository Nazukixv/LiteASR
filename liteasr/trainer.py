"""Trainer."""

from contextlib import nullcontext
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

        train_set = self.task.dataset("train").batchify(
            "train",
            self.cfg.dataset,
            self.cfg.postprocess,
        )
        valid_set = self.task.dataset("valid").batchify(
            "valid",
            self.cfg.dataset,
            self.cfg.postprocess,
        )

        if self.cfg.distributed.world_size > 1:
            train_sampler = DistributedSampler(train_set)
            valid_sampler = DistributedSampler(valid_set)
        else:
            train_sampler = None
            valid_sampler = None

        self.train_iter = EpochDataLoader(
            dataset=train_set,
            batch_size=1,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=lambda x: x[0],
        )
        self.valid_iter = DataLoader(
            dataset=valid_set,
            batch_size=1,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=lambda x: x[0],
        )

        self.device = torch.device("cuda")

        self._add_events()
        self.loss = 0

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

    def _add_events(self):
        trigger_store = {}
        for t in self.cfg.common.trigger:
            trigger_store[t.name] = Trigger(t.interval, t.unit)

        self.event_manager = EventManager()
        for key in trigger_store:
            if hasattr(self, key):
                event = trigger_store[key](getattr(self, key))
                self.event_manager.add_event(event)

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
        for i, batch in enumerate(self.train_iter, start=1):
            # trigger epoch-wise events
            self.event_manager.trigger_epoch_events(self)

            # stop training process if reach limit
            if self.stop():
                break

            # train one step
            batch = to_device(batch, self.device)

            if (
                self.cfg.distributed.world_size > 1
                and i % self.cfg.optimization.accum_grad != 0
            ):
                sync_context = self.model.no_sync
            else:
                sync_context = nullcontext

            with sync_context():
                loss = self.criterion(self.model, *batch)
                self.loss += loss / self.cfg.optimization.accum_grad
                loss.backward()

            if i % self.cfg.optimization.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.optimization.clip_grad_norm,
                )
                if not math.isnan(grad_norm):
                    self.optimizer.step()
                    self.iter += 1

                    # trigger iteration-wise events
                    self.event_manager.trigger_iteration_events(self)
                else:
                    if self.is_master():
                        logger.warning(
                            "iteration {} is skipped since gradient is NaN".
                            format(self.iter + 1)
                        )

                self.optimizer.zero_grad()
                self.loss = 0

    def report_loss(self):
        if self.cfg.distributed.world_size > 1:
            dist.reduce(self.loss, dst=0)
            self.loss /= self.cfg.distributed.world_size
        logger.info(
            "{} / {} iters, {} / {} epochs - current loss: {:.2f}".format(
                self.iter,
                self.max_iter,
                self.epoch,
                self.max_epoch,
                self.loss.item(),
            )
        )

    def valid(self):
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

    def save_model(self):
        if self.is_master():
            model_name = "model.ep.{}.pt".format(self.epoch)
            self.task.save_model(model_name, self.model)

    def inference(self):
        if self.is_master():
            self.model.eval()
            with torch.no_grad():
                for test_set in self.task.dataset("test"):
                    for data in test_set:
                        feat = data.x.unsqueeze(0).cuda()
                        ref = data.text
                        hyp = self.task.inference(feat, self.model)
                        res = "[X]" if ref == hyp else "[ ]"
                        logger.debug(f"{res} {hyp}")
            self.model.train()
