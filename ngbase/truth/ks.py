import numpy as np
import scipy

from ngbase.truth.utils import get_interpolated_gt


def get_true_ks_fs(path='./outputs/truth/gt_ks_200.mat'):

    data = scipy.io.loadmat(path)
    t = np.float32(data['t'][0])
    x = np.float32(data['x'][0])
    usol = np.float32(data['Uvals'])

    gt_f = get_interpolated_gt(usol, (t, x))
    return [gt_f]
