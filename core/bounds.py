import numpy as np
import torch

from core.kl_inv import klInvFunction
from core.utils import find_ns
from models.majority_vote import MultipleMajorityVote


def mcallester_bound(n, model, risk, delta, coeff=1, verbose=False, monitor=None):

    kl = model.KL()

    if isinstance(model, MultipleMajorityVote): # informed priors
        
        const = np.log(4 * (n**2 / 4)**0.5 / delta)
        kl *= 2

    else:
        const = np.log(2 * (n**0.5) / delta)

    bound = coeff * (risk + ((kl + const) / 2 / n)**0.5)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 


def seeger_bound(n, model, risk, delta, div, sample=False, coeff=1, order=None, verbose=False, monitor=None):

    if sample:
        kl = model.KL_dis()
    elif div == 'KL':
        kl = model.KL()
    elif div == 'Renyi':
        kl = model.Renyi(order)

    if isinstance(model, MultipleMajorityVote): # informed priors
        
        const = np.log(4 * (n**2 / 4)**0.5 / delta)
        kl *= 2

    else:
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * n**0.5)
        else:
            const = np.log(2 * (n**0.5) / delta)

    if coeff == 0:
        bound = risk
    elif sample:
        bound = coeff * klInvFunction.apply(risk, torch.max(kl + const, torch.tensor(0)) / n)
    else:
        bound = coeff * klInvFunction.apply(risk, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound


def triple_bound(n, model, risks, delta, div, sample=False, coeff=1, order=None, return_single=False, verbose=False, monitor=None):
    ns = torch.tensor(find_ns(risks, n))

    if sample:
        kl = model.KL_dis()
    elif div == 'KL':
        kl = model.KL()
    elif div == 'Renyi':
        kl = model.Renyi(order)

    if isinstance(model, MultipleMajorityVote):  # informed priors

        consts = np.log(4 * (ns ** 2 / 4) ** 0.5 / (delta / 3))
        kl *= 2

    else:
        if div == 'Renyi':
            consts = (2 * order - 1) / (order - 1) * np.log(2 / (delta / 3)) + np.log(2 * ns ** 0.5)
        else:
            consts = np.log(2 * (ns ** 0.5) / (delta / 3))

    if sample:
        bound_1 = coeff * klInvFunction.apply(risks[0], torch.max(kl + consts[0], torch.tensor(0)) / ns[0], "MAX")
        bound_2 = coeff * klInvFunction.apply(risks[1], torch.max(kl + consts[1], torch.tensor(0)) / ns[1], "MIN")
        bound_3 = coeff * klInvFunction.apply(risks[2], torch.max(kl + consts[2], torch.tensor(0)) / ns[2], "MIN")
    else:
        bound_1 = coeff * klInvFunction.apply(risks[0], (kl + consts[0]) / ns[0], "MAX")
        bound_2 = coeff * klInvFunction.apply(risks[1], (kl + consts[1]) / ns[1], "MIN")
        bound_3 = coeff * klInvFunction.apply(risks[2], (kl + consts[2]) / ns[2], "MIN")
    bound = (bound_1 - bound_3) / (bound_2 - bound_3)

    if verbose:
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": (risks[0].item(), risks[1].item(), risks[2].item())})

    if return_single:
        return bound
    return torch.tensor([bound_3, bound_2])


BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound,
    "triple": triple_bound,
}
