import numpy as np
import torch

from core.kl_inv import klInvFunction
from core.utils import find_ns, bin_cum
from models.majority_vote import MultipleMajorityVote


def test_set_bound(n_err, n, delta):
    """
    Implementation of Langford's test bound (Theorem 3.3, Tutorial on Practical Prediction Theory for Classification).
    """
    gamma_sup, gamma_inf, gamma = 1, 0, 0.5
    for j in range(10):
        pro = bin_cum(n_err, n, gamma)
        if pro >= delta:
            gamma_inf = gamma
        else:
            gamma_sup = gamma
        gamma = (gamma_sup + gamma_inf) / 2
    return gamma

def vcdim_bound(n, model, err, delta):
    """
    Implementation of the VC-dim generalization bound.
    """
    vc_dim = torch.tensor(model.get_post().shape)
    return (err + ((vc_dim * (np.log(2 * n / vc_dim) + 1) + np.log(4 / delta)) / n) ** 0.5).item()

def seeger_bound(n, model, risk, delta, div, disintegrated=False, coeff=1, order=None, verbose=False, monitor=None):
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
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * n)
        else:
            const = np.log(2 * n / delta)
    else:
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * n ** 0.5)
        else:
            const = np.log(2 * (n ** 0.5) / delta)

    if coeff == 0:
        bound = 1
    else:
        # Possible cases where kl + const < 0 for disintegrated bounds, thus max().
        bound = coeff * klInvFunction.apply(risk, torch.max(kl + const, torch.tensor(0)) / n)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound


def triple_bound(n, model, risks, delta, div, disintegrated=False, coeff=1, order=None, return_single=False, verbose=False, monitor=None):
    """
    Implementation of the triple bound (see paper).
    """
    # See the def. of find_ns.
    ns = torch.tensor(find_ns(risks, n))
    delta_div = delta / 3

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
            consts = (2 * order - 1) / (order - 1) * np.log(2 / delta_div) + np.log(2 * ns)
        else:
            consts = np.log(4 * (ns ** 2 / 4) ** 0.5 / delta_div)
    else:
        if div == 'Renyi':
            consts = (2 * order - 1) / (order - 1) * np.log(2 / delta_div) + np.log(2 * ns ** 0.5)
        else:
            consts = np.log(2 * (ns ** 0.5) / delta_div)

    # Possible cases where kl + const < 0 for disintegrated bounds, thus max().
    bound_1 = coeff * klInvFunction.apply(risks[0], torch.max(kl + consts[0], torch.tensor(0)) / ns[0], "MAX")
    bound_2 = coeff * klInvFunction.apply(risks[1], torch.max(kl + consts[1], torch.tensor(0)) / ns[1], "MIN")
    bound_3 = coeff * klInvFunction.apply(risks[2], torch.max(kl + consts[2], torch.tensor(0)) / ns[2], "MIN")
    bound = (bound_1 - bound_3) / (bound_2 - bound_3)

    if verbose:
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": (risks[0].item(), risks[1].item(), risks[2].item())})

    if return_single:
        return bound
    return torch.tensor([bound_3, bound_2])


BOUNDS = {
    "seeger": seeger_bound,
    "triple": triple_bound,
    "test": test_set_bound,
    "vcdim": vcdim_bound
}
