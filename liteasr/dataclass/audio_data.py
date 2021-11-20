from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Audio(object):
    uttid: str
    fd: str
    start: Union[int, float]
    end: Union[int, float]
    shape: List[int]
    tokenids: Optional[List[int]] = None
