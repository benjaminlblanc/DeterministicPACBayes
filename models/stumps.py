import torch
import numpy as np

## DECISION STUMPS ##
# Only supports binary classification

def uniform_decision_stumps(M, d, min_v, max_v, init):
    """
    Create the decision stump-forest classifier, composed of 2*M evenly spaced stumps per feature.
        The first M predicts a class, the other one the other class.
    M (int): number of stumps per feature dimension for a single class prediction (out of two).
    d (int): number of feature dimension.
    min_v, max_v (floats): min and max values of every feature.
    initialization (): type of stump initialization.

    Returns the collection of stumps and the number of base classifier (2 * M * d).
    """
    assert type(M) == int, f"M must be an integer, got {M}."
    assert init in ['ones', 'rand'], f"init (cfg.model.stump_init) must be in ['ones', 'rand'], got {init}."

    # Get M evenly spaced thresholds in the interval [min_v, max_v] per dimension
    thresholds = torch.from_numpy(np.linspace(min_v, max_v, M, endpoint=False, axis=-1)).float()

    # Two possible initializations
    if init == 'ones':
        sigs = torch.ones((d, M * 2))
    else:
        sigs = torch.rand((d, M * 2))

    sigs[..., :M] *= -1  # first M*d stumps return one class, last M*d return the other
    stumps = lambda x: stumps_predict(x, torch.cat((thresholds, thresholds), 1), sigs)
    return stumps, d * M * 2

def stumps_predict(x, thresholds, signs):
    return (signs * (1 - 2*(x[..., None] > thresholds))).reshape((len(x), -1))