import numpy as np
import torch

from core.kl_inv import klInvFunction
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

def seeger_bound(n, model, risk, delta, div, coeff=1, order=None, verbose=False, monitor=None):

    if div == 'KL':
        kl = model.KL()
    elif div == 'Renyi':
        kl = model.Renyi(order)
    elif div == 'KL_dis':
        kl = model.KL_dis()

    if isinstance(model, MultipleMajorityVote): # informed priors
        
        const = np.log(4 * (n**2 / 4)**0.5 / delta)
        kl *= 2

    else:
        if div == 'Renyi':
            const = (2 * order - 1) / (order - 1) * np.log(2 / delta) + np.log(2 * n**0.5)
        else:
            const = np.log(2 * (n**0.5) / delta)

    bound = coeff * klInvFunction.apply(risk, (kl + const) / n)

    if verbose:
        print(f"Empirical risk={risk.item()}, KL={kl}, const={const}, n={n}")
        print(f"Bound={bound.item()}\n")

    if monitor:
        monitor.write(train={"KL": kl.item(), "risk": risk.item()})

    return bound 


BOUNDS = {
    "mcallester": mcallester_bound,
    "seeger": seeger_bound,
}
