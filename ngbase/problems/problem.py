from dataclasses import dataclass
from typing import Callable, List

import jax.numpy as jnp


@dataclass
class Problem:
    """
    Problem is a Class which organizes everything needed to specify a PDE of interest.
    In this sense a single instance of a Problem should uniquely map to some PDE solution

    Attributes
    ----------
    name : str
        string identifier for problem
    dim : int
        number of spatial dimensions
    quantities : int
        number of quantities
    ics : List[List[Callable]]
        2D list of Callable functions. 
        Dims are q x d where d is number of derivatives to fit and q is tne number of quantities
        functions should be of form init_condition(X), where X is the spatial domain
    omega_init : ndarray
        specifies region from which to sample points to solve for initial conditions
    omega : ndarray
        specifies region from which to sample points for primary intergration
    rhsides : List[Callable]
        list of RHS for each quantity of the given system
    period: float = None
        length of the peroid of the problem, will be enforced via non-linear paramterization
        should be length of full domain in each dim
    zero_bc: float = None
        zero boundary condition should be length of full domain in each dim
    """

    dim: int
    quantities: int
    derivatives: int

    ics: List[List[Callable]]

    omega_init: jnp.ndarray
    omega: jnp.ndarray
    rhsides: List[Callable]
    name: str = ''
    period: float = None
    zero_bc: float = None
