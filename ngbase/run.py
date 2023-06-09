import argparse
import os
import random
import secrets
from pathlib import Path
from pprint import pformat
from time import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.lib.xla_bridge import get_backend
from tqdm import tqdm

from ngbase.dnn.ansatz import Ansatz
from ngbase.dnn.get import get_ansatz
from ngbase.io.store import (RESULT, convert_results_to_dict, load_init,
                             save_init, save_results)
from ngbase.misc.misc import timer, unique_id
from ngbase.ops.ops import get_ops
from ngbase.problems.get import get_problem
from ngbase.sample.data_sampler import get_data_sampler
from ngbase.solvers.get_init import get_init
from ngbase.solvers.timeODE import timeODE
from ngbase.solvers.timeSGD import timeSGD
from ngbase.truth.get import get_truth
from ngbase.truth.utils import get_grid, get_relative_error, get_solutions

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".35"


@timer
def run(args) -> dict:

    #########
    # Setup #
    #########

    # make id
    RESULT['id'] = unique_id(5)
    print(f'id: {RESULT["id"]}')

    if args.x64:
        jax.config.update("jax_enable_x64", True)
        print('enabling x64')

    if args.platform is not None:
        jax.config.update('jax_platform_name', args.platform)
    # if oop error see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    print(
        f'platform: {get_backend().platform} — device_count: {jax.local_device_count()}')

    # set jit_store
    output_dir = Path(args.output_dir)

    # log arguments
    print(args.__dict__)

    # set random seed, if none use random random seed
    if args.seed == -1:
        args.seed = secrets.randbelow(10_000)
        print(f'seed: {args.seed}')
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    random.seed(seed)
    np.random.seed(seed)

    key_init = jax.random.PRNGKey(args.seed_init)

    rcond = args.rcond
    if rcond == 0.0:
        rcond = None

    # get setup objects
    problem = get_problem(args.problem)
    ansatz = get_ansatz(args.model, args.activation, args.width,
                        args.depth, problem.period, problem.zero_bc, key_init)
    sampler = get_data_sampler(args.sampler, problem.omega,
                               problem.omega_init, args.batch_size, args.batch_size_init)

    # init ansatz
    theta_init = ansatz.init_ansatz(sampler.init_pt)
    total_params = len(theta_init)
    RESULT['total_params'] = total_params
    print(f'{total_params} total_params')

    sub_params = args.sub_params
    if args.method == 'opt_dis_sub':
        if sub_params > total_params:
            print('sub_params exiting...')
            return
        if sub_params == 1 or sub_params == -1:
            sub_params = total_params
        RESULT['sub_params'] = sub_params
        args.sub_params = sub_params
        print(f'update {sub_params}/{total_params} params')

    ops = get_ops(args.method, problem, ansatz,
                  rcond, sub_params, args.sub_sampler, args.sub_sampler_n, args.ls_rank)

    dt, Tend = args.dt, args.Tend
    t_eval = np.linspace(0.0, Tend, int(Tend/dt)+1)
    print(f'{len(t_eval)} time steps')
    if args.t_pts is None:
        interval = 1
    else:
        interval = max(len(t_eval)//args.t_pts, 1)
    t_store = t_eval[::interval]

    ###############################
    # Solve for inital conditions #
    ###############################
    if args.load_init is not None:
        print('loading init conditions...')
        theta_0s, loss_history, final_losses = load_init(
            args.load_init, args)
        print(f'init losses: {final_losses}')
        if args.x64:
            theta_0s = [jnp.asarray(t, dtype=jnp.float64) for t in theta_0s]
    else:
        print('fitting init cond...')
        theta_0s, loss_history, final_losses = get_init(problem, sampler, ansatz, theta_init,
                                                        args.solver_iters_init, args.solver_lr_init, args.solver_optimizer_init, args.solver_lbfgs_i_init, args.solver_tol_init, args.solver_fit_ds_init, key_init)

    if args.save_init:
        print('saving init conditions...')
        save_init((theta_0s, loss_history, final_losses),  args)
    if args.only_init:
        print('only_init exiting...')
        return

    RESULT['init_final_losses'] = final_losses

    ############################
    # Perform time integration #
    ############################
    time_int = time()
    # perform integration
    if 'opt_dis' in args.method:
        time_thetas = timeODE(ops, sampler=sampler, t_eval=t_eval, t_store=t_store, theta_0s=theta_0s, method=args.method,
                              scheme=args.scheme, ansatz=ansatz, seed=seed)

    elif args.method == 'dis_opt':
        time_thetas = timeSGD(
            ops, ansatz=ansatz, sampler=sampler, problem=problem, t_eval=t_eval, t_store=t_store, theta_0s=theta_0s, key=key,
            iters=args.solver_iters, lr=args.solver_lr, scheduler=args.solver_scheduler, optimizer=args.solver_optimizer,
            lbfgs_i=args.solver_lbfgs_i,  corrector=args.solver_corrector, tol=args.solver_tol)

    time_took = time() - time_int
    RESULT['time_int'] = time_took

    print('finished solve!')

    ##############################################
    # Compute ground truth, error, store results #
    ##############################################
    with jax.default_device(jax.devices("cpu")[0]):
        omega = np.asarray(problem.omega, dtype=np.float32)
        t_store = np.asarray(t_store, dtype=np.float32)

        # process results
        time_thetas = np.split(time_thetas, problem.quantities, axis=1)

        # transform from thetas back to physical space, if sol_pts is not None
        if args.sol_pts is not None:
            solution_grid = get_grid(omega, args.sol_pts)
            solution = get_solutions(ansatz, time_thetas, solution_grid)
            RESULT['solution'] = solution
        print('eval sol!')

        truth_fs = get_truth(problem, Tend)
        if truth_fs is not None:
            time_space_grid = get_grid(omega, args.sol_pts, time=t_store)
            rl, rl_time = get_relative_error(
                truth_fs, solution, time_space_grid)
            RESULT['relative_error_time'] = rl_time
            RESULT['relative_error'] = rl

            for q, e in enumerate(rl):
                print(f'relative_error q{q}: {e:.2E}')

        # organize results
        RESULT['dim'] = problem.dim
        RESULT['quantities'] = problem.quantities
        RESULT['thetas'] = time_thetas
        RESULT['t_eval'] = t_eval
        RESULT['t_store'] = t_store

        # save
        result_dict = convert_results_to_dict(RESULT, args)

        save_results(result_dict, args.output_dir, args.output_name)

        print('done!')

    return result_dict


def get_parser():
    parser = argparse.ArgumentParser()

    ####################################
    # PREQUIRED, NO DEAFULT #
    ####################################
    parser.add_argument("--problem", "-p", help="problem name", type=str)
    parser.add_argument("--method", "-m", help="method name",
                        choices=['dis_opt', 'opt_dis', 'opt_dis_sub'], type=str)
    parser.add_argument("--dt", "-t", help="time-step size", type=float)
    parser.add_argument(
        "--Tend", "-T", help="end time of integration", type=float)

    ##################
    # ARGS FOR SOLVE #
    ##################
    parser.add_argument("--model", help="model", type=str, default='dnn')
    parser.add_argument("--width", help="number of nodes",
                        type=int, default=25)
    parser.add_argument("--depth", help="depth of network",
                        type=int, default=3)
    parser.add_argument(
        "--activation", help="activation function for dnn", type=str, default='rational')
    parser.add_argument("--sampler", help="how to sample data",  type=str,
                        choices=["equi", "uni"], default='equi')
    parser.add_argument(
        "--batch_size", help="number of points to sample on interior domain", type=int, default=1_000)

    parser.add_argument("--seed", help="random seed, if -1 seed will be randomized (i.e. not reproducable)",
                        type=int, default=1)
    parser.add_argument(
        "--seed_init", help="seed to for init, and for loading init", type=int, default=1)

    ###################
    # ARGS FOR INIT_C #
    ###################
    parser.add_argument(
        "--solver_lbfgs_i_init", help="number of lbfgs iterations to start", type=int, default=500)
    parser.add_argument(
        "--solver_lr_init", help="solver to use for inital conditions", type=float, default=1e-3)
    parser.add_argument(
        "--batch_size_init", help="number of points to sample on interior domain", type=int, default=10_000)
    parser.add_argument(
        "--solver_iters_init", help="max number of iterations for init_c solver", type=int, default=10_000)
    parser.add_argument(
        "--solver_optimizer_init", help="which optimizer to use", type=str, default='adam')
    parser.add_argument(
        "--solver_tol_init", help="loss tolerance for init solver", type=float, default=None)
    parser.add_argument(
        "--solver_fit_ds_init", help="how many deriviatves to fit of init_c, if None will do all avaiable", type=int, default=None)
    parser.add_argument(
        "--save_init", help="whether to save init condition", type=bool_force, default=False)
    parser.add_argument(
        "--load_init", help="whether to load init condition, use 'auto' or otherwise pass path to file", type=str, default=None)
    parser.add_argument(
        "--only_init", help="whether to quit after save init condition", type=bool_force, default=False)

    ####################
    # ARGS FOR OPT_DIS #
    ####################
    parser.add_argument("--scheme", help="name of time integration scheme", type=str, default='rk4')
    parser.add_argument(
        "--sub_params", help="subset of parameters to update, if -1 set to total params", type=int, default=-1)
    parser.add_argument(
        "--sub_sampler", help="method for sampling subset of parameters, [rand, norm, score]", type=str, default='rand')
    parser.add_argument(
        "--sub_sampler_n", help="resample ever n times", type=int, default=4)
    parser.add_argument(
        "--ls_rank", help="rank for leverage score approximation", type=int, default=None)
    parser.add_argument(
        "--rcond", help="rcond param for lstsq solve in opt_dis scheme", type=float,  default=1e-4)

    ####################
    # ARGS FOR DIS_OPT #
    ####################
    parser.add_argument(
        "--solver_lr", help="adam learning rate", type=float, default=1e-5)
    parser.add_argument(
        "--solver_iters", help="max number of iterations for init_c solver", type=int, default=10_000)
    parser.add_argument(
        "--solver_scheduler", help="whether to use learning rate scheduler", type=bool_force, default=True)
    parser.add_argument(
        "--solver_optimizer", help="which optimizer to use", type=str, default='adam')
    parser.add_argument(
        "--solver_lbfgs_i", help="number of lbfgs iterations to start", type=int, default=500)
    parser.add_argument(
        "--solver_corrector", help="number of corrector steps", type=int, default=0)
    parser.add_argument(
        "--solver_tol", help="resiudal threshold for early stopping", type=float, default=None)

    #################
    # ARGS FOR MISC #
    #################
    parser.add_argument(
        "--x64", help="whether to use 64 bit precision in jax", type=bool_force, default=False)
    parser.add_argument(
        "--platform", help="gpu or cpu, None will let jax default", type=str, default=None)
    parser.add_argument(
        "--output_dir", help="where to save results, if None nothing is saved", type=str, default='./outputs')
    parser.add_argument(
        "--output_name", help="name of results file, if None, name is auto generated", type=str, default=None)
    parser.add_argument(
        "--sol_pts", help="number of points on the solution grid", type=int,  default=10_000)
    parser.add_argument(
        "--t_pts", help="number of points of time points to store theta, if None will store all", type=int,  default=None)
    return parser


def bool_force(x): return (str(x).lower() == 'true')


if __name__ == "__main__":

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    parser = get_parser()
    args = parser.parse_args()
    results = run(args)
