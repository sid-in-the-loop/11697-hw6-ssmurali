import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass

class Embedding:
    def __init__(self, raw):
        self.vector: Optional[np.ndarray] = None

        if isinstance(raw, np.ndarray):
            self.from_numpy(raw)
        elif isinstance(raw, torch.Tensor):
            self.from_tensor(raw)
        elif isinstance(raw, list):
            self.from_list(raw)
        else:
            raise TypeError(f"Cant create Embedding from object of type {type(raw)}")
    
    def from_numpy(self, nparray):
        self.vector = nparray

    def from_tensor(self, tensor):
        # assuming tensor -> numpy will not be a bottleneck for us
        # this will only be called when generating embeddings
        # and we wont generate new embeddings too often
        self.vector = tensor.detach().cpu().numpy()

    def from_list(self, l):
        self.vector = np.array(l)

    def get_vector(self) -> np.ndarray:
        assert self.vector is not None
        return self.vector

@dataclass
class Chunk:
    id: int
    doc_id: int
    string: str
    embedding: Embedding

    def __post_init__(self):
        #print(f"new chunk of type {type(self.embedding.get_vector())}")
        pass


