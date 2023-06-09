import jax
import numpy as np
import optax
from jax import jit
from tqdm.auto import tqdm

str_to_opt = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'amsgrad':  optax.amsgrad,
    'adabelief': optax.adabelief
}


def adam_opt(theta_init, loss, args, init_state=None, steps=500, learning_rate=1e-3, scheduler=True, verbose=False, loss_tol=None, optimizer='adam', key=None, return_p_hist=False):
    # adds warm up cosine decay
    if scheduler:
        learning_rate = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=0.0
        )

    opti_f = str_to_opt[optimizer]
    optimizer = opti_f(learning_rate=learning_rate)

    state = optimizer.init(theta_init)
    if init_state is not None:
        state = init_state

    @jit
    def step(params, state, args):
        loss_value, grads = jax.value_and_grad(loss)(params, *args)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return loss_value, params, state

    losses = np.zeros(steps)
    params = theta_init
    opt_loss, opt_params, opt_state = float('inf'), None, None
    pbar = tqdm(range(steps), disable=not verbose)

    if return_p_hist:
        params_hist = np.zeros((steps, *params.shape))

    for i in pbar:
        if return_p_hist:
            params_hist[i] = params

        cur_args = args
        if callable(args):
            cur_args, key = args(key)
        loss, params_new, state_new = step(params, state, cur_args)
        losses[i] = loss
        if loss < opt_loss:
            opt_params = params
            opt_loss = loss
            opt_state = state
            pbar.set_postfix({'loss': f'{opt_loss:.2E}'})

        params = params_new
        state = state_new

        if loss_tol is not None and opt_loss < loss_tol:
            break

    if return_p_hist:
        return opt_params, opt_loss, opt_state, losses, params_hist

    return opt_params, opt_loss, opt_state, losses
