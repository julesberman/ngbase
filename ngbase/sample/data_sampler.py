import abc
from typing import List, Tuple

import jax
import jax.numpy as jnp

from ngbase.sample.sampler_helper import (sample_equidistant_helper,
                                          sample_uniform_helper,
                                          sample_uniform_helper_focus)


class DataSampler(abc.ABC):

    def __init__(self, name, omega, omega_init, batch_size=None, batch_size_init=None, iters=None, learning_rate=None):
        self.name = name
        self.omega = omega
        self.omega_init = omega_init
        self.batch_size = batch_size
        self.batch_size_init = batch_size_init
        self.init_pt = jnp.zeros(omega.shape[-1])
        self.iters = iters
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def sample_data() -> Tuple[List[jnp.ndarray], jax.random.PRNGKey]:
        raise NotImplementedError()

    @abc.abstractmethod
    def sample_data_init() -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        raise NotImplementedError()

    def get_batch_size(self, batch_size):
        if batch_size is None:
            batch_size = self.batch_size
        return batch_size

    def get_batch_size_init(self, batch_size):
        if batch_size is None:
            batch_size = self.batch_size_init
        return batch_size


class UniformDataSampler(DataSampler):
    def __init__(self, *args, **kwargs):
        DataSampler.__init__(self, *args, **kwargs)

    def sample_data(self, key, batch_size=None):
        batch_size = self.get_batch_size(batch_size)
        return sample_uniform_helper(self.omega, batch_size, key)

    def sample_data_init(self, key, batch_size=None):
        batch_size = self.get_batch_size_init(batch_size)
        return sample_uniform_helper_focus(self.omega, self.omega_init, batch_size, key)


class EquidistantDataSampler(DataSampler):
    def __init__(self, *args, **kwargs):
        DataSampler.__init__(self, *args, **kwargs)

    def sample_data(self, key, batch_size=None):
        batch_size = self.get_batch_size(batch_size)
        samples = sample_equidistant_helper(self.omega, batch_size)
        return samples, key

    def sample_data_init(self, key, batch_size=None):
        batch_size = self.get_batch_size_init(batch_size)
        samples = sample_equidistant_helper(self.omega, batch_size)
        return samples, key


def get_data_sampler(samplerName: str, omega: jnp.ndarray,  omega_init: jnp.ndarray, batch_size: int, batch_size_init: int) -> DataSampler:
    creator = None
    if (samplerName == "uni"):
        creator = UniformDataSampler
    elif (samplerName == "equi"):
        creator = EquidistantDataSampler
    else:
        raise Exception(f"Unknown Sampling Scheme: {samplerName}")

    return creator(samplerName, omega, omega_init, batch_size=batch_size, batch_size_init=batch_size_init)
