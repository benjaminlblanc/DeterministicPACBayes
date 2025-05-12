import torch
import numpy as np
import random

from betaincder import betainc, betaincderp, betaincderq
from torch import lgamma, log1p, exp, log
from torch.special import erf


def whether_to_run_run(cfg):
    if cfg.training.distribution in "dirichlet":
        assert cfg.model.prior in ["adjusted", 1]
    elif cfg.training.distribution in "categorical":
        assert cfg.model.prior == "adjusted"
    elif cfg.training.distribution == "gaussian":
        assert cfg.model.prior == 0


def updating_first_seed_results(seed_results, cfg, time, model, train_err, test_err, best_train_stats, deterministic_bound, ben_bound_no_finetune):
    seed_results["train-error"] = train_err['error']
    seed_results["test-error"] = test_err['error']
    seed_results["train-risk"] = best_train_stats["error"]
    seed_results["deterministic_bound"] = deterministic_bound
    seed_results["ben_bound_no_finetune"] = ben_bound_no_finetune
    seed_results["time"] = time
    seed_results["posterior"] = model.get_post().detach().numpy()
    seed_results["strength"] = best_train_stats["strength"]
    seed_results["KL"] = model.KL().item()
    seed_results["entropy"] = model.entropy().item()
    if ben_bound_no_finetune != 1:
        seed_results["factor_no_finetune"] = ben_bound_no_finetune / deterministic_bound * 2
    return seed_results

def updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune, i):
    seed_results["seed"] = cfg.training.seed+i
    seed_results["train-error_finetune"] = train_error['error']
    seed_results["test-error_finetune"] = test_error['error']
    seed_results["ben_bound_with_finetune"] = ben_bound_with_finetune
    if ben_bound_with_finetune != 1:
        seed_results["factor_with_finetune"] = ben_bound_with_finetune / seed_results["deterministic_bound"] * 2
    return seed_results

def deterministic(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def betaincderx(x, a, b):
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    partial_x = exp((b - 1) * log1p(-x) + (a - 1) * log(x) - lbeta)
    return partial_x

def Phi(z):
    """
    Computes the Phi function.
    """
    return 1 / 2 * (1 - erf(z / 2 ** 0.5))

class BetaInc(torch.autograd.Function):
    """ regularized incomplete beta function and its forward and backward passes"""

    @staticmethod
    def forward(ctx, p, q, x, order):

        x = torch.clamp(x, 0, 1)

        ctx.save_for_backward(p, q, x, order)
        # deal with dirac distributions
        if p == 0.:
            return torch.tensor(1.) # for any x, cumulative = 1.

        elif q == 0. or x == 0.:
            return torch.tensor(0.) # cumulative = 0.
    
        return torch.tensor(betainc(x, p, q) ** order.item())

    @staticmethod
    def backward(ctx, grad):
        p, q, x, order = ctx.saved_tensors
        
        if p == 0. or q == 0. or x == 0.: # deal with dirac distributions
            grad_p, grad_q, grad_x, grad_order = 0., 0., 0., 0.

        else:
            grad_p, grad_q, grad_x, grad_order = betaincderp(x, p, q), betaincderq(x, p, q), betaincderx(x, p, q), torch.zeros(1)

        return grad * grad_p, grad * grad_q, grad * grad_x, grad * grad_order
