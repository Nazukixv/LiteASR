"""Initialize sub package."""

import importlib
import os

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from liteasr.config import LiteasrDataclass

CRITERION_REGISTRY = {}
CRITERION_DATACLASS_REGISTRY = {}
CRITERION_CLASS_NAMES = set()


class LiteasrLoss(object):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def build_criterion(self, cfg, task):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self._loss(*input, **kwargs)


def build_criterion(cfg, task) -> LiteasrLoss:
    criter_name = getattr(cfg, "name", None)

    criterion = CRITERION_REGISTRY[criter_name]

    dc = CRITERION_DATACLASS_REGISTRY[criter_name]
    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)

    return criterion.build_criterion(merged_cfg, task)


def register_criterion(name, dataclass=None):

    def register_criterion_cls(cls):
        CRITERION_REGISTRY[name] = cls
        CRITERION_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None:
            assert issubclass(dataclass, LiteasrDataclass)
            CRITERION_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="criterion", node=node)

        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/ directory
criterions_dir = os.path.dirname(__file__)
for file in os.listdir(criterions_dir):
    path = os.path.join(criterions_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        criter_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("liteasr.criterions." + criter_name)
