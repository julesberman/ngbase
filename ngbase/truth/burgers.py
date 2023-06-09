import numpy as np
import scipy

from ngbase.truth.utils import get_interpolated_gt


def get_true_burgers_fs(path='./outputs/truth/gt_burgers_10.mat'):

    data = scipy.io.loadmat(path)
    t = data['t'][0]
    x = data['x'][0]
    usol = data['Uvals']

    gt_f = get_interpolated_gt(usol, (t, x))
    return [gt_f]
