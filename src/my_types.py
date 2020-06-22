import torch
from torch import Tensor
from typing import Dict, Optional, Tuple

Batch = Tuple[Tensor, Tensor]
Device = Optional[torch.device]

FloatDict = Dict[str, float]

LossWithStats = Tuple[Tensor, FloatDict]
