
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


def true_vlasov2s(problem, t_eval, T,  Nx, Nv, n_store=1, fwd_euler=False):

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
    N = int(len(X)**(1/2))
    x_period = c-a
    N = int(len(X)**(1/2))
    v_space = X.reshape(Nx, Nv, 2)[0, :, 1]
    x_space = X.reshape(Nx, Nv, 2)[:, 0, 0]
    dx = x_space[1] - x_space[0]
    dy = v_space[1] - v_space[0]

    # Modes for Poisson equation
    modes = np.zeros(Nx)
    k = 2 * np.pi / x_period
    modes[:Nx//2] = k * np.arange(Nx//2)
    modes[Nx//2:] = - k * np.arange(Nx//2, 0, -1)
    modes += modes == 0
    modes = jnp.array(modes)

    def enforce_bcs(u):
        # periodic BCs
        u = u.at[bc_x1].set(u[bc_x2])
        u = u.at[bc_v1].set(u[bc_v2])
        return u

    def compute_E(integral_f):
        rhok = jnp.fft.fft(integral_f)/modes
        return jnp.real(jnp.fft.ifft(-1j*rhok))

    def get_E(f_e, f_i):

        f_e = f_e.reshape(Nx, Nv)
        f_i = f_i.reshape(Nx, Nv)
        integral_f = trapz(f_i - f_e, v_space)

        E = compute_E(integral_f)
        e_energy = jnp.sqrt(0.5*trapz(jnp.abs(E)**2, x_space))

        # this is essentially interpolation but
        # NOTE once again assuming equi spaced grid
        E = jnp.repeat(E[:, jnp.newaxis], Nv, axis=1)
        E = E.reshape(len(X))

        return E, e_energy

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
    def rhs(t, y, eps=1.0):
        f_e, f_i = jnp.split(y, 2)

        f_e = enforce_bcs(f_e)
        f_i = enforce_bcs(f_i)

        E, e_energy = get_E(f_e, f_i)

        # id_tap(_save_E, e_energy)

        grad_fe = get_grad(f_e, dx, dy)
        grad_fi = get_grad(f_i, dx, dy)

        dx_fe = grad_fe[0].ravel()
        dv_fe = grad_fe[1].ravel()

        dx_fi = grad_fi[0].ravel()
        dv_fi = grad_fi[1].ravel()

        x, v = X[:, 0], X[:, 1]

        dt_fe = -(1/eps)*v*dx_fe+(1/eps)*E*dv_fe
        dt_fi = -v*dx_fi-E*dv_fi

        return jnp.concatenate([dt_fe, dt_fi])

    ic_p_f, ic_b_f = problem.ics[0][0], problem.ics[1][0]
    ic_p = ic_p_f(X)
    ic_v = ic_b_f(X)
    ic = jnp.concatenate([ic_p, ic_v])

    #     result = scipy.integrate.solve_ivp(
    #         rhs, (t_eval[0], t_eval[-1]), ic, method='RK45', t_eval=t_store, max_step=dt)
    #     y = result.y

    if fwd_euler:
        y = odeint_fe(rhs, ic, t_eval)
    else:
        y = odeint_rk4(rhs, ic, t_eval)
    y = jnp.swapaxes(y, 0, 1)

    y_p, y_v = np.split(y, 2, axis=0)
    sol_p = np.transpose(y_p.reshape(Nx, Nv, -1), axes=[2, 0, 1])
    sol_v = np.transpose(y_v.reshape(Nx, Nv, -1), axes=[2, 0, 1])

    return [sol_p[::n_store], sol_v[::n_store]], [t_eval[::n_store], x_space, v_space]


def grid_from_spacing(spacing):
    m_grids = jnp.meshgrid(*spacing, indexing='ij')
    m_grids = [m.flatten() for m in m_grids]
    X = jnp.array(m_grids).T
    return X


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    problem = get_problem('vlasov2s')

    T = 60.0
    dt = 5e-3
    n_store = 2
    t_store = np.linspace(0.0, T, int(T/dt)+1)
    Nx = 500
    Nv = 1500
    print(t_store)
    print(t_store[::n_store])
    sols, spacing = true_vlasov2s(problem, t_store, T, Nx, Nv, n_store=n_store)
    print('solved!')

    sols = [np.asarray(s, dtype=np.float32) for s in sols]
    spacing = [np.asarray(s, dtype=np.float32) for s in spacing]

    data = {'true': sols, 'spacing': spacing}

    # f_p = get_interpolated_gt(sols[0], spacing)
    # f_v = get_interpolated_gt(sols[1], spacing)

    outpath = Path(f'/scratch/jmb1174/truth/gt_{problem.name}_60.pkl')

    save_pickle(outpath, data)
    print('done!')
