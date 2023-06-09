import glob
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.experimental.host_callback import id_print, id_tap

# class Result:
#     jit_store: bool = True
#     id: str = None
#     dim: int = None
#     quantities: int = None
#     ansatz: Ansatz = None
#     X_samples: np.ndarray = None
#     t_eval: np.ndarray = None
#     t_store: np.ndarray = None
#     residuals: np.ndarray = None
#     total_params: int = None
#     sub_params: int = None
#     cols_take: np.ndarray = None
#     pi: np.ndarray = None
#     time_int: float = None

#     # computed from grid
#     grid: np.ndarray = None

#     # per quantity
#     thetas: np.ndarray = None
#     u_0_true: np.ndarray = None
#     solution: List[np.ndarray] = None
#     relative_error_time: np.ndarray = None
#     relative_error: float = None
#     time_loss: float = None
#     time_err: np.ndarray = None
#     time_sol_err: np.ndarray = None
#     time_true_err: np.ndarray = None
#     init_final_losses: np.ndarray = None

#     residuals: np.ndarray = None
#     rank: np.ndarray = None
#     s_vals: np.ndarray = None

#     # storage
#     aux: List[Any] = None


RESULT = {}


def jit_save(data, key: str):
    """
    jit_save allows a user to save some data object into the RESULT object within a jitted function
    this is possible through the callback function id_tap which allows one to execute a function on the host machine within a jitted function
    autmoatically the data object is accumlated into a python List in the RESULT object, this makes it natural to call jit_save in a jitted loop

    Parameters
    ----------
    data : Any
        data object to be saved into RESULT
    key : str
        the location of the data object in RESULT is given by key
    """

    def save_on_host(data, transforms):
        if not key in RESULT:
            RESULT[key] = []

        RESULT[key].append(data)

    id_tap(save_on_host, data)


def convert_list_to_numpy(dic: dict):
    for key, value in dic.items():
        if isinstance(value, list) and key != 'aux':
            dic[key] = np.array(value)
    return dic


def convert_jax_to_numpy(dic: dict):
    for key, value in dic.items():
        if isinstance(value, jnp.ndarray):
            dic[key] = np.array(value)
    return dic


def convert_results_to_dict(results, args):

    r_dict = results
    # needed to convert device arrays
    r_dict = convert_jax_to_numpy(r_dict)
    # needed to convert python list from jit_save
    r_dict = convert_list_to_numpy(r_dict)

    # convert args to dict
    args_d = vars(args)
    all_data = {**args_d, **r_dict}

    return all_data


_SEP_ = '-'


def get_initc_file_str(args):
    args_d = vars(args)
    output_fields = ['problem', 'activation',
                     'model', 'depth', 'width', 'solver_lr_init', 'seed_init']
    output_fields = [str(args_d[f]).replace('.', '_') for f in output_fields]
    output_name = _SEP_.join(output_fields)
    output_name = f'initc{_SEP_}{output_name}'

    return output_name


def load_init(load_init_str, args):
    # load_init_path = f'/scratch/jmb1174/{args.problem}/ic'
    load_init_path = f'./outputs/ics/{args.problem}_init/'
    load_init_path = Path(load_init_path)
    if load_init_str == 'auto':
        output_name = get_initc_file_str(args)
        output_path = (load_init_path / output_name).with_suffix(".pkl")
    else:
        output_path = load_init_path / load_init_str

    init = pd.read_pickle(output_path)

    return (init['thetas'], init['loss_history'], init['final_losses'])


def save_init(inits, args, output_dir=None):
    if output_dir is None:
        output_dir = Path(f'./outputs/ics/{args.problem}_init')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_name = get_initc_file_str(args)

    output_path = (output_dir / output_name).with_suffix(".pkl")
    theta_0s, loss_history, final_losses = inits
    inits_d = {'thetas': theta_0s, 'loss_history': loss_history,
               'final_losses': final_losses}
    args_d = vars(args)
    all_data = {**args_d, **inits_d}
    try:
        with open(output_path, "wb") as outfile:
            pickle.dump(all_data, outfile)
    except Exception as e:
        print(e)
        print(f"could not save to {output_path}")


def save_results(data: dict, output_dir: str, output_name: str):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if output_name is None:
        output_fields = ['problem', 'method', 'dt', 'model', 'id']
        output_fields = [str(data[f]).replace('.', '_') for f in output_fields]
        output_name = _SEP_.join(output_fields)

    output_path = (output_dir / output_name).with_suffix(".pkl")
    try:
        with open(output_path, "wb") as outfile:
            pickle.dump(data, outfile)
        print(f'results saved to {output_path.absolute()}')
    except Exception as e:
        print(e)
        print(f"could not save to {output_path}")


def consolidate_db(db_dir, remove_old=True):

    all_data = []
    i = 0
    pkl_files = glob.glob(os.path.join(db_dir, "*.pkl"))
    for file_path in pkl_files:
        with open(file_path, "rb") as f:
            file_name = os.path.basename(file_path)
            print(file_name)
            data = pickle.load(f)
            all_data.append(data)
            i += 1

    dataframe = pd.DataFrame(all_data)
    dataframe_file = (db_dir / "db").with_suffix(".pkl")
    db_dir.mkdir(exist_ok=True, parents=True)
    print(db_dir)
    print(dataframe_file)
    if os.path.exists(dataframe_file):
        print(f"{dataframe_file} exists; appending to dataframe")
        old_df = pd.read_pickle(dataframe_file)
        dataframe = pd.concat([old_df, dataframe])

    dataframe.to_pickle(dataframe_file)
    print(f"{i} files consolidated to {dataframe_file}")
    if remove_old:
        i = 0
        for filename in pkl_files:
            os.remove(filename)
            i += 1
        print(f"{i} files removed")

    return dataframe


if __name__ == "__main__":

    for p in ['fix_n_vary_s_wave']:
        db_dir = Path(f'./outputs/{p}')
        db = consolidate_db(db_dir, remove_old=False)
        print(db)
        print('done!')
