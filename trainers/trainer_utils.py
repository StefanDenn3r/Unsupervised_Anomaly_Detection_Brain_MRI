import numpy as np

from utils.utils import normalize


def get_summary_dict(batch, run, visualization_keys=None, *others):
    if visualization_keys is None:
        visualization_keys = ['reconstruction', 'L1']
    run = dict(filter(lambda x: x[1] is not None, run.items()))
    visuals = np.asarray([
        255 * np.hstack([
            normalize(batch[i]),
            *[normalize(run[key][i]) for key in visualization_keys],
            *[normalize(element[i]) for element in others]
        ]) for i in range(len(batch))]
    )
    scalars = dict(filter(lambda x: not (type(x[1]) == float and x[1] != x[1]) and x[1].ndim == 0, run.items()))
    return scalars, visuals
