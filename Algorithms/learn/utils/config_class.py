import torch as th
import torch.nn as nn
import yaml

from typing import Optional, Any, Tuple, List, Dict

class Config:
    def __init__(self,
                 env_kwargs: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 learn_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):

        self.env_kwargs = env_kwargs
        self.model_kwargs = model_kwargs
        self.learn_kwargs = learn_kwargs
        self.kwargs = kwargs

    def default(self):
        pass

    def update(self, key1):