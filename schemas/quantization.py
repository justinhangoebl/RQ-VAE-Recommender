from typing import NamedTuple
from torch import Tensor

class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor