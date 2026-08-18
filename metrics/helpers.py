# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Decorators used by the metric suite.
# Adapted from utils3d (https://github.com/EasternJournalist/utils3d), MIT License.
import torch
from torch import Tensor
from numbers import Number
import inspect
from typing import *
from functools import wraps


__all__ = [
    'toarray',
    'batched',
]

P = ParamSpec("P")  
R = TypeVar("R")


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper


class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs
    
    def __call__(self, fn: Callable[[P], R]) -> Callable[[P], R]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)
        return wrapper  
    
    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)



def totensor(
    *args_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str, None], 
    _others: Union[torch.dtype, str] = None, 
    **kwargs_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that converts non-array arguments to array of specified default dtype.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dtypes_dict = {
            **dict(zip(argnames, args_dtypes)),
            **kwargs_dtypes
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            inputs = {
                **{argnames[i]: x for i, x in enumerate(args)},
                **kwargs
            }
            if len(input_devices := tuple(x.device for x in inputs.values() if isinstance(x, Tensor))) > 0:
                device = input_devices[0]
            else:
                device = None
            args = tuple(
                torch.tensor(x).to(device, getattr(inputs[dtype], 'dtype', None) if isinstance(dtype, str) else dtype)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype := dtypes_dict.get(argnames[i], _others)) is not None \
                else x
                for i, x in enumerate(args)
            )
            kwargs = {
                k: torch.tensor(x).to(device, getattr(inputs[dtype], 'dtype', None) if isinstance(dtype, str) else dtype)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype := dtypes_dict.get(k, _others)) is not None \
                else x
                for k, x in kwargs.items()
            }
            return func(*args, **kwargs)
        return wrapper
    return decorator


def batched(*args_dims: Union[int, None], _others: Union[int, None] = None, **kwargs_dims: Union[int, None]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that extends a function's input and out batch dimensions.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dims_dict = {
            **dict(zip(argnames, args_dims)),
            **kwargs_dims
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            args = list(args)
            # Get arguments non-batch dimensions
            args_dim = tuple(dims_dict.get(argname, _others) for argname in argnames[:len(args)])
            kwargs_dim = {k: dims_dict.get(k, _others) for k in kwargs}
            # Find the common batch shape
            batch_shape = torch.broadcast_shapes(*(
                x.shape[:x.ndim - dim] 
                for x, dim in zip((*args, *kwargs.values()), (*args_dim, *kwargs_dim.values())) 
                if isinstance(x, Tensor) and dim is not None
            ))
            # Broadcast and flatten batch dimensions
            args = tuple(
                torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and dim is not None else x
                for x, dim in zip(args, args_dim)
            )
            kwargs = {
                k: torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and (dim := kwargs_dim[k]) is not None else x
                for k, x in kwargs.items()
            }
            # Call function
            result = func(*args, **kwargs)
            # Restore batch shape
            if isinstance(result, tuple):
                result = tuple(
                    x.reshape((*batch_shape, *x.shape[1:])) if isinstance(x, Tensor) else x
                    for x in result
                )
            elif isinstance(result, Tensor):
                result = result.reshape((*batch_shape, *result.shape[1:]))
            return result
        return wrapper
    return decorator
