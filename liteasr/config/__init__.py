"""Initialize sub package."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LiteasrDataclass(object):
    name: Optional[str] = None


@dataclass
class LiteasrConfig(LiteasrDataclass):
    common: Any = None
    distributed: Any = None
    task: Any = None
    model: Any = None
    criterion: Any = None
    optimizer: Any = None
