import numpy as np
import scipy

from ngbase.truth.utils import get_interpolated_gt


def get_true_bz_fs(path='./outputs/truth/gt_bz_5.mat'):

    data = scipy.io.loadmat(path)
    t = data['t'][0]
    x = data['x'][0]
    usol1 = data['Uvals1']
    usol2 = data['Uvals2']
    usol3 = data['Uvals3']

    gt_f1 = get_interpolated_gt(usol1, (t, x))
    gt_f2 = get_interpolated_gt(usol2, (t, x))
    gt_f3 = get_interpolated_gt(usol3, (t, x))
    return [gt_f1, gt_f2, gt_f3]
