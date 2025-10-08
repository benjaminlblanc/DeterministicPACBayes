import numpy as np
import torch

from core.kl_inv import klInvFunction
from core.utils import bin_cum
from models.majority_vote import MultipleMajorityVote


def test_set_bound(m_err, m, delta):
    """
    Implementation of Langford's test bound (Theorem 3.3, Tutorial on Practical Prediction Theory for Classification).
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(10):
        pro = bin_cum(m_err, m, gamma)
        if pro >= delta:
            gamma_inf = gamma
        else:
            gamma_sup = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma

def vcdim_bound(m, model, err, delta):
    """
    Implementation of the VC-dim generalization bound.
    """
    vc_dim = torch.tensor(model.get_post().shape)
    return (err + ((vc_dim * (np.log(2 * m / vc_dim) + 1) + np.log(4 / delta)) / m) ** 0.5).item()

def seeger_bound(m, model, risk, delta, div, disintegrated=False, coeff=1, order=None, verbose=False, monitor=None):
    """
    Implementation of Seeger's PAC-Bayes bound.
    """
    # We gather the divergence penalty depending on the situation
    if div == 'Renyi':
        kl = model.Renyi(order)
    elif disintegrated:
        kl = model.KL_dis()
    elif div == 'KL':
        kl = model.KL()

    # We compute the constant depending on the predictor type and the bound type
    if isinstance(model, MultipleMajorityVote): # informed priors
        kl *= 2
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * m)
        else:
            const = np.log(2 * m / delta)
    else:
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * m ** 0.5)
        else:
            const = np.log(2 * (m ** 0.5) / delta)

    if coeff == 0:
        bound = 1
    else:
        # Possible cases where kl + const < 0 for disintegrated bounds, thus max().
        bound = coeff * klInvFunction.apply(risk, torch.max(kl + const, torch.tensor(0)) / m)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, m={m}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound


BOUNDS = {
    "test": test_set_bound,
    "vcdim": vcdim_bound,
    "seeger": seeger_bound
}
