"""Initialize sub package."""

import importlib
import os

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch.nn as nn

from liteasr.config import LiteasrDataclass

MODEL_REGISTRY = {}  # model_name -> model_cls
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}  # arch_name  -> model_cls
ARCH_MODEL_NAME_REGISTRY = {}  # arch_name  -> model_name
ARCH_MODEL_INV_REGISTRY = {}  # model_name -> [arch_name]
ARCH_CONFIG_REGISTRY = {}  # arch_name  -> arch_fn


class LiteasrModel(nn.Module):

    def __init__(self):
        super().__init__()

    def build_model(cls, cfg, task):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    def script(self):
        import torch
        _self = torch.jit.script(self)
        return _self

    def get_pred_len(self, xlens):
        raise NotImplementedError

    def get_target_len(self, ylens):
        raise NotImplementedError


def build_model(cfg, task) -> LiteasrModel:
    model_name = getattr(cfg, "name", None)

    model = MODEL_REGISTRY[model_name]

    dc = MODEL_DATACLASS_REGISTRY[model_name]
    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)

    return model.build_model(merged_cfg, task)


def register_model(name, dataclass=None):

    def register_model_cls(cls):
        MODEL_REGISTRY[name] = cls

        if dataclass is not None:
            assert issubclass(dataclass, LiteasrDataclass)
            MODEL_DATACLASS_REGISTRY[name] = dataclass
            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node)

        return cls

    return register_model_cls


# maybe deprecated?
def register_model_architecture(model_name, arch_name):

    def register_model_arch_fn(fn):
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_architecture


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_") and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[:file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("liteasr.models." + model_name)
