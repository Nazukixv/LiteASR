"""Initialize sub package."""

import importlib
import os

from hydra.core.config_store import ConfigStore

from liteasr import criterions
from liteasr import models
from liteasr import optims
from liteasr.config import LiteasrDataclass
from liteasr.criterions import LiteasrLoss
from liteasr.models import LiteasrModel
from liteasr.optims import LiteasrOptimizer

# register dataclass
TASK_DATACLASS_REGISTRY = {}
TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


class LiteasrTask(object):

    def __init__(self):
        self._train_data = []
        self._valid_data = []
        self.train_set = None
        self.valid_set = None

    def load_data(self, cfg):
        raise NotImplementedError

    def inference(self, x, model: LiteasrModel):
        raise NotImplementedError

    def save_model(self, model_name: str, model: LiteasrModel):
        raise NotImplementedError

    def build_model(self, cfg) -> LiteasrModel:
        model = models.build_model(cfg, self)
        return model

    def build_optimizer(self, params, cfg) -> LiteasrOptimizer:
        optimizer = optims.build_optimizer(params, cfg, self)
        return optimizer

    def build_criterion(self, cfg) -> LiteasrLoss:
        criterion = criterions.build_criterion(cfg, self)
        return criterion

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for key in sorted(self.__dict__.keys()):
            format_string += '\n'
            format_string += '  ' + key + ': ' + str(self.__dict__[key])
        format_string += '\n)'
        return format_string


def setup_task(cfg: LiteasrDataclass) -> LiteasrTask:
    task_name = getattr(cfg, "name", None)
    task = TASK_REGISTRY[task_name]
    return task(cfg)


def register_task(name, dataclass=None):

    def register_task_cls(cls):
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None:
            assert issubclass(dataclass, LiteasrDataclass)
            TASK_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="task", node=node)

        return cls

    return register_task_cls


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith("_") and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        task_name = file[:file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("liteasr.tasks." + task_name)
