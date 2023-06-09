import argparse
import time
from itertools import product
from pathlib import Path

import submitit
from tqdm import tqdm

from ngbase.io.store import consolidate_db
from ngbase.run import get_parser, run

# config = {
#     'problem': ['vlasov2s'],
#     'method': ['dis_opt'],
#     'dt': [1e-2],
#     'Tend': [60.0],
#     'seed': [1],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'depth': [1, 2, 3, 4, 5],
#     'width': [25],
#     'save_init': [True],
#     'batch_size_init': [50_000],
#     'solver_iters_init': [2_000_000],
#     'solver_lr_init': [1e-3],
#     'solver_lbfgs_i_init': [1000],
#     'batch_size': [25_000],
#     'solver_iters': [10_000],
#     'solver_optimizer': ['adam'],
#     'solver_scheduler': ['True'],
#     'solver_lr': [5e-6],
#     'solver_lbfgs_i': [500],
#     'solver_corrector': [1],
#     'solver_use_init': ['False'],
#     'sampler': ['equi'],
# }


config = {
    'problem': ['vlasov2s'],
    'method': ['opt_dis'],
    'dt': [1e-2],
    'Tend': [60.0],
    'seed': [1],
    'depth': [2, 4, 5],
    'width': [25],
    'model': ['dnn'],
    'activation': ['rational'],
    'batch_size': [20_000],
    'sampler': ['equi'],
    'solver_iters': [5000],
    'solver_optimizer': ['adam'],
    'solver_scheduler': ['True'],
    'solver_lr': [1e-5],
    'rcond': [1e-6, 5e-7, 1e-7],
    'solver_lbfgs_i': [500],
    'solver_corrector': [1],
    'load_init': ['auto'],
}


# config = {
#     'problem': ['wavebc'],
#     'method': ['dis_opt'],
#     'dt': [1e-2],
#     'Tend': [8.0],
#     'seed': [1],
#     'depth': [1, 2, 3, 4, 5],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [25_000],
#     'sampler': ['equi'],
#     'load_init': ['auto'],
#     'solver_iters': [10_000],
#     'solver_optimizer': ['adam'],
#     'solver_scheduler': ['True'],
#     'solver_lr': [5e-5, 1e-5],
#     'solver_lbfgs_i': [500],
#     'solver_corrector': [1],
#     'load_init': ['auto'],
# }


# config = {
#     'problem': ['kdv'],
#     'method': ['dis_opt', 'opt_dis'],
#     'dt': [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4],
#     'Tend': [4.0],
#     'seed': [1],
#     'depth': [2],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'sampler': ['equi'],
#     'load_init': ['auto'],
#     'solver_iters': [10_000],
#     'solver_optimizer': ['adam'],
#     'solver_scheduler': ['True'],
#     'solver_lr': [1e-5],
#     'solver_lbfgs_i': [500],
#     'solver_corrector': [1],
#     'load_init': ['auto'],
#     'rcond': [1e-7]
# }

# config = {
#     'problem': ['vlasovfix'],
#     'method': ['dis_opt'],
#     'dt': [1e-2],
#     'Tend': [4.0],
#     'seed': [1],
#     'depth': [1, 2, 3, 4, 5, 6, 7, 8],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'sampler': ['equi'],
#     'save_init': [True],
#     'only_init': [True],
#     'batch_size_init': [50_000],
#     'solver_iters_init': [1_000_000],
#     'solver_lr_init': [1e-3],
#     'solver_lbfgs_i_init': [1000],
# }


# config = {
#     'problem': ['bz'],
#     'method': ['dis_opt', 'opt_dis'],
#     'dt': [4e-2*16, 4e-2*8, 4e-2*4, 4e-2*2, 4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3],
#     # 'dt': [4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3],
#     'Tend': [4.0],
#     'seed': [1],
#     'depth': [2],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'sampler': ['equi'],
#     'load_init': ['auto'],
#     'solver_iters': [10_000],
#     'solver_optimizer': ['adam'],
#     'solver_scheduler': ['True'],
#     'solver_lr': [1e-5],
#     'solver_lbfgs_i': [500],
#     'solver_corrector': [1],
#     'load_init': ['auto'],
#     'rcond': [1e-7]
# }


# config = {
#     'problem': ['vlasov2s'],
#     'method': ['dis_opt'],
#     'dt': [5e-3],
#     'Tend': [80.0],
#     'seed': [1],
#     'model': ['dnn'],
#     'depth': [5],
#     'width': [30],
#     'batch_size': [20_000],
#     'sampler': ['equi'],
#     'solver_iters': [15_000],
#     'solver_scheduler': [True],
#     'solver_optimizer': ['adam'],
#     'solver_lr': [5e-5, 1e-5, 5e-6],
#     'load_init': ['auto'],
#     't_pts': [1000],
# }


# config = {
#     'problem': ['wavebc'],
#     'method': ['opt_dis_rand'],
#     'dt': [1e-2],
#     'Tend': [6.0],
#     'seed': [1],
#     'depth': [2, 4],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'random_p': [25, 50, 100, 200, 400, 800, 1600, 3200],
#     'rcond': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
#     'load_init': ['auto'],
#     'x64': [True, False]
# }

# config = {
#     'problem': ['kdv'],
#     'method': ['opt_dis_rand'],
#     'dt': [1e-4],
#     'Tend': [3.0],
#     'seed': [7],
#     'depth': [1, 2, 3, 4],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [4_000],
#     'sampler': ['equi'],
#     'load_init': ['auto'],
#     'sub_params': [25, 50, 100, 200, 400, 800, 1200, 1600],
#     'rcond': [5e-5],
# }

# config = {
#     'problem': ['burgers'],
#     'method': ['opt_dis_rand'],
#     'dt': [5e-3],
#     'Tend': [4.0],
#     'seed': [7],
#     'depth': [1, 2, 3, 4, 5, 6],
#     'width': [25],
#     'model': ['dnn'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'sampler': ['equi'],
#     'load_init': ['auto'],
#     'sub_params': [25, 50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 1],
#     'rcond': [1e-4]
# }


# config = {
#     'problem': ['burgers'],
#     'method': ['opt_dis_rand'],
#     'dt': [5e-3],
#     'Tend': [4.0],
#     'seed': [8],
#     'depth': [2, 3, 4, 5, 6],
#     'width': [25],
#     'model': ['dnnsin'],
#     'activation': ['rational'],
#     'batch_size': [10_000],
#     'sampler': ['equi'],

#     'sub_params': [50, 100, 200, 400, 800, 1200, 1600, 2400, 3200, 1],
#     'rcond': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0],

#     'batch_size_init': [25_000],
#     'solver_iters_init': [500_000],
#     'solver_lr_init': [1e-3],
#     'solver_lbfgs_i_init': [500],
#     'solver_fit_ds_init': [1],
#     'solver_tol_init': [1e-4],
# }


config = {
    'problem': ['ac'],
    'method': ['opt_dis_rand'],
    'dt': [1e-2],
    'Tend': [5.0],
    'seed_init': [15],
    'depth': [1, 2, 3, 4],
    'width': [25],
    'model': ['dnn'],
    'load_init': ['auto'],
    'activation': ['rational'],
    'batch_size': [10_000],
    'sampler': ['equi'],
    'sub_params': [100, 200, 400, 800, 1200, 1600, 2400, 3200, 1],
    'rcond': [1e-4],
}


# config = {
#     'problem': ['ac'],
#     'method': ['dis_opt'],
#     'dt': [1e-2],
#     'Tend': [4.0],
#     'seed_init': [15, 16],
#     'depth': [1, 2, 3, 4],
#     'width': [25],
#     'model': ['dnn',],
#     'activation': ['rational'],
#     'sampler': ['equi'],
#     'save_init': [True],
#     'only_init': [True],
#     'batch_size_init': [50_000],
#     'solver_iters_init': [1_000_000],
#     'solver_lr_init': [1e-3],
#     'solver_lbfgs_i_init': [1000],
# }


def submit(submit_args):

    print(submit_args.__dict__)

    exp_dir = Path(submit_args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    experiment_name = '-'.join(config['problem'])

    # set up slurm
    slurm_dir = exp_dir / "slurm"
    slurm_dir.mkdir(exist_ok=True)
    executor = submitit.AutoExecutor(folder=slurm_dir)
    # executor.update_parameters(
    #     name=experiment_name[:8],
    #     mem_gb=100,
    #     cpus_per_task=2,
    #     timeout_min=60 * 20,  # less than 2 days # max is 60 * 72
    #     slurm_array_parallelism=256,
    #     tasks_per_node=1,
    #     nodes=1,
    #     # you can choose to comment this, or change it to v100 as per your need
    #     slurm_gres=f"gpu:{submit_args.gres}",
    #     slurm_signal_delay_s=120,
    # )

    executor.update_parameters(
        name=submit_args.exp_dir[-8:],
        mem_gb=100,
        timeout_min=1200,
        cpus_per_task=4,
        gpus_per_node=1,
        slurm_partition="gpu",
        slurm_array_parallelism=256,
    )

    # executor.update_parameters(
    #     name=experiment_name[:8],
    #     mem_gb=80,
    #     timeout_min=1200,
    #     cpus_per_task=24,
    #     slurm_partition="ccn",
    #     slurm_array_parallelism=256,
    # )

    # get argparser for run function
    parser = get_parser()

    # take cartesian product of all options and produce args object for each
    all_args_lists = list(product(*config.values()))
    keys = [f'--{k}' for k in config.keys()]
    jobs = []
    n_jobs = len(jobs)
    print(f"you are launching {len(all_args_lists)} jobs 😅")
    for a_l in all_args_lists:
        keys_args = zip(keys, a_l)
        flat = [str(item) for sublist in keys_args for item in sublist]

        args = parser.parse_args(flat)

        # set defaults args
        args.output_dir = exp_dir

        # submit to slurm
        job = executor.submit(run, args)
        jobs.append(job)

    print(f"waiting for jobs to finish, then consolidate")
    pbar = tqdm(total=n_jobs)
    prev_finished, num_finished = 0, 0
    while num_finished != n_jobs:
        num_finished = sum(job.done() for job in jobs)
        if num_finished > prev_finished:
            pbar.update(num_finished - prev_finished)
            prev_finished = num_finished
        time.sleep(5.0)

    # write all db files into one dataframe
    if submit_args.consolidate:
        dataframe = consolidate_db(exp_dir, remove_old=False)
        print("print consolidated db\n", dataframe)

    print("submit done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp_dir", "-ed", help="path to save experiment", type=str, required=True)
    parser.add_argument(
        "--consolidate", help="whether to consolidate all files into db after", type=bool, default=False)
    parser.add_argument(
        "--gres", help="number of gpus to request", default=1)

    args = parser.parse_args()
    results = submit(args)
