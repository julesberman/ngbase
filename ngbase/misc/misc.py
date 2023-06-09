
import os
import random
import string
from functools import wraps
from time import time

import jax.numpy as jnp
from jax.experimental.host_callback import id_print, id_tap
from tqdm.auto import tqdm


def timer(func):
    """
    Function Timer decorator
    use as:

    @timer
    def my_func():
        ....
        return 

    Will print: "Function my_func executed in 3.0242s"
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def jqdm(total, argnum=0, decimals=1, **kwargs):
    "Decorate a jax scan body function to include a TQDM progressbar."

    pbar = tqdm(range(100), mininterval=500, **kwargs)

    def _update(cur, transforms):
        amt = float(cur*100/total)
        amt = round(amt, decimals)
        if amt != pbar.last_print_n:
            pbar.n = amt
            pbar.last_print_n = amt
            pbar.refresh()

    def update_jqdm(cur):
        id_tap(_update, cur),

    def _jqdm(func):

        @wraps(func)
        def wrapper_body_fun(*args, **kwargs):
            cur = args[argnum]
            update_jqdm(cur)
            result = func(*args, **kwargs)
            return result  # close_tqdm(result, amt)

        return wrapper_body_fun

    return _jqdm


def get_cpu_count() -> int:
    cpu_count = None
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = len(os.sched_getaffinity(0))
            return cpu_count
        except:
            pass

    cpu_count = os.cpu_count()
    if cpu_count is not None:
        return cpu_count

    try:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
    except:
        pass

    print("could not get cpu count, returning 1")

    return 1


def unique_id(n: int = 10) -> str:
    """creates unique alphanumeric id w/ low collision probability"""
    chars = string.ascii_letters + string.digits
    id = "".join(random.choice(chars) for _ in range(n))
    return id


def relative_error(true=None, test=None, eps=1e-5, **kwargs):
    assert test is not None
    assert true is not None
    norm = jnp.linalg.norm(true, **kwargs)
    if norm < eps:
        norm += 1.0
    return jnp.linalg.norm(test-true, **kwargs) / norm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')
