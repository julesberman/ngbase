

import pandas as pd

from ngbase.truth.ac import get_true_ac_fs
from ngbase.truth.burgers import get_true_burgers_fs
from ngbase.truth.bz import get_true_bz_fs
from ngbase.truth.kdv import get_true_kdv_fs
from ngbase.truth.ks import get_true_ks_fs
from ngbase.truth.utils import get_interpolated_gt


def get_truth(problem, t_end):
    try:
        if problem.name == "ac":
            assert t_end <= 10.0, f'ac gt computed up to T=10.0, given: {t_end}'
            gt_fs = get_true_ac_fs()
        elif problem.name == "burgers":
            assert t_end <= 10.0, f'ac gt computed up to T=10.0, given: {t_end}'
            gt_fs = get_true_burgers_fs()
        elif problem.name == "ks":
            gt_fs = get_true_ks_fs()
        elif problem.name == "bz":
            assert t_end <= 5.0, f'bz gt computed up to T=5.0, given: {t_end}'
            gt_fs = get_true_bz_fs()
        elif problem.name == "wavebc":
            assert t_end <= 8.0, f'wavebc gt computed up to T=8.0, given: {t_end}'
            gt_fs = load_gt('./outputs/truth/gt_wavebc_8.pkl')
        elif problem.name == "vlasov2s":
            assert t_end <= 60.0, f'vlasov2s gt computed up to T=60.0, given: {t_end}'
            gt_fs = load_gt('./outputs/truth/gt_vlasov2s_60.pkl')
        elif problem.name == "vlasovfix":
            gt_fs = load_gt('./outputs/truth/gt_vlasovfix_3.pkl')
        elif problem.name == "kdv":
            gt_fs = get_true_kdv_fs()
        else:
            return None
    except:
        print('error loading truth returning None')
        return None
    return gt_fs


def load_gt(path):
    dic = pd.read_pickle(path)
    us, spacing = dic['true'], dic['spacing']
    f_s = [get_interpolated_gt(u, spacing) for u in us]
    return f_s
