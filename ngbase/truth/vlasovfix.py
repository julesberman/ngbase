

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.numpy import trapz

from ngbase.io.utils import save_pickle
from ngbase.misc.misc import jqdm
from ngbase.problems.get import get_problem
from ngbase.solvers.odeint import odeint_fe, odeint_rk4


def true_vlasovfix(problem, t_eval, T,  Nx, Nv, n_store=1, fwd_euler=False):

    (a, b), (c, d) = problem.omega

    x = jnp.linspace(a, c, Nx)
    v = jnp.linspace(b, d, Nv)
    m_grids = jnp.meshgrid(x, v, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = jnp.array(m_grids).T

    (a, b), (c, d) = problem.omega
    bc_v1 = (X[:, 0] == a)
    bc_v2 = (X[:, 0] == c)

    bc_x1 = (X[:, 1] == d)
    bc_x2 = (X[:, 1] == b)
    v_space = X.reshape(Nx, Nv, 2)[0, :, 1]
    x_space = X.reshape(Nx, Nv, 2)[:, 0, 0]
    dx = x_space[1] - x_space[0]
    dy = v_space[1] - v_space[0]

    def enforce_bcs(u):
        # periodic BCs
        u = u.at[bc_x1].set(u[bc_x2])
        u = u.at[bc_v1].set(u[bc_v2])
        return u

    def get_grad(u, dx, dy):
        u = u.reshape(Nx, Nv)

        u_ext = jnp.zeros((Nx+4, Nv+4))
        u_ext = u_ext.at[2:-2, 2:-2].set(u)

        u_ext = u_ext.at[2:-2, 0].set(u[:, -3])
        u_ext = u_ext.at[2:-2, 1].set(u[:, -2])

        u_ext = u_ext.at[0, 2:-2].set(u[-3, :])
        u_ext = u_ext.at[1, 2:-2].set(u[-2, :])

        u_ext = u_ext.at[2:-2, -1].set(u[:, 2])
        u_ext = u_ext.at[2:-2, -2].set(u[:, 1])

        u_ext = u_ext.at[-1, 2:-2].set(u[2, :])
        u_ext = u_ext.at[-2, 2:-2].set(u[1, :])

        # 4th order central difference
        grad_x = (1/(dx))\
            * (1 * u_ext[0:-4, 2:-2]
                - 8 * u_ext[1:-3, 2:-2]
                + 8 * u_ext[3:-1, 2:-2]
                - 1 * u_ext[4:, 2:-2]
               ) / 12
        grad_y = (1/(dy))\
            * (1 * u_ext[2:-2, 0:-4]
                - 8 * u_ext[2:-2, 1:-3]
                + 8 * u_ext[2:-2, 3:-1]
                - 1 * u_ext[2:-2, 4:]
               ) / 12

        grad_f = jnp.stack([grad_x, grad_y])
        return grad_f

    @jit
    @jqdm(T)
    def rhs(t, f):

        f = enforce_bcs(f)

        grad_f = get_grad(f, dx, dy)

        dx_fi = grad_f[0].ravel()
        dv_fi = grad_f[1].ravel()

        x, v = X[:, 0], X[:, 1]
        E = -jnp.sin(x)
        dt_f = -v*dx_fi+E*dv_fi

        return dt_f

    ic_f = problem.ics[0][0]
    ic = ic_f(X)

    if fwd_euler:
        y = odeint_fe(rhs, ic, t_eval)
    else:
        y = odeint_rk4(rhs, ic, t_eval)
    y = jnp.swapaxes(y, 0, 1)

    sol_p = np.transpose(y.reshape(Nx, Nv, -1), axes=[2, 0, 1])

    return [sol_p[::n_store]], [t_eval[::n_store], x_space, v_space]


def grid_from_spacing(spacing):
    m_grids = jnp.meshgrid(*spacing, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = jnp.array(m_grids).T
    return X


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    problem = get_problem('vlasovfix')

    T = 3.0
    dt = 1e-3
    n_store = 1
    t_store = np.linspace(0.0, T, int(T/dt)+1)
    Nx = 1024
    Nv = 1024
    print(t_store)
    print(t_store[::n_store])
    sols, spacing = true_vlasovfix(
        problem, t_store, T, Nx, Nv, n_store=n_store)
    print('solved!')

    sols = [np.asarray(s, dtype=np.float32) for s in sols]
    spacing = [np.asarray(s, dtype=np.float32) for s in spacing]

    data = {'true': sols, 'spacing': spacing}

    outpath = Path(f'./outputs/truth/gt_{problem.name}_3.pkl')

    save_pickle(outpath, data)
    print('done!')
