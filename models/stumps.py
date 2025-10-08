import torch
import numpy as np

## DECISION STUMPS ##
# Only supports binary classification

def uniform_decision_stumps(n, d, min_v, max_v, init, distr):
    """
    Create the decision stump-forest classifier, composed of 2*n evenly spaced stumps per feature.
        The first n predicts a class, the other one the other class.
    n (int): number of stumps per feature dimension for a single class prediction (out of two).
    d (int): number of feature dimension.
    min_v, max_v (floats): min and max values of every feature.
    initialization (): type of stump initialization.

    Returns the collection of stumps and the number of base classifier (2 * n * d).
    """
    assert type(n) == int, f"n must be an integer, got {n}."
    assert init in ['ones', 'rand'], f"init (cfg.model.stump_init) must be in ['ones', 'rand'], got {init}."

    # Get n evenly spaced thresholds in the interval [min_v, max_v] per dimension
    thresholds = torch.from_numpy(np.linspace(min_v, max_v, n, endpoint=False, axis=-1)).float()

    if distr == "gaussian":
        stumps = lambda x: stumps_predict(x, thresholds, 1)
        return stumps, d * n
    else:
        # Two possible initializations
        if init == 'ones':
            sigs = torch.ones((d, n * 2))
        elif init == 'rand':
            sigs = torch.rand((d, n * 2))
            sigs[..., n:] = sigs[..., :n]

        sigs[..., :n] *= -1  # first n*d stumps return one class, last n*d return the other
        stumps = lambda x: stumps_predict(x, torch.cat((thresholds, thresholds), 1), sigs)
        return stumps, d * n * 2

def stumps_predict(x, thresholds, signs):
    return (signs * (1 - 2*(x[..., None] > thresholds))).reshape((len(x), -1))