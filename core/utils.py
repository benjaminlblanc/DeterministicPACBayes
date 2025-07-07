import torch
import numpy as np
import random

from scipy.stats import multivariate_normal
from betaincder import betainc, betaincderp, betaincderq
from torch import lgamma, log1p, exp, log
from torch.special import erf


def whether_to_run_run(cfg):
    if cfg.training.distribution in "dirichlet":
        assert cfg.model.prior in ["adjusted", 1]
        assert cfg.model.stump_init == "ones"
        assert cfg.training.risk != "Dis_Renyi"
    elif cfg.training.distribution in "categorical":
        assert cfg.model.prior == "adjusted"
        assert cfg.model.stump_init == "rand"
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order
    elif cfg.training.distribution == "gaussian":
        assert cfg.model.prior == 0
        assert cfg.model.stump_init == "rand"
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order < 2

    if cfg.training.risk == "Bin":
        assert cfg.training.rand_n > 0



def updating_first_seed_results(seed_results, time, model, train_err, test_err, deterministic_bound, final_bound, ben_bound_no_finetune):
    seed_results["train-error"] = train_err['error']
    seed_results["test-error"] = test_err['error']
    seed_results["test-error_sampled"] = test_err['error_sampled']
    seed_results["test-error_sampled_std"] = test_err['error_sampled_std']
    seed_results["deterministic_bound"] = deterministic_bound
    seed_results["deterministic_bound_sampled"] = final_bound["bound_sampled"]
    seed_results["deterministic_bound_sampled_std"] = final_bound["bound_sampled_std"]
    seed_results["ben_bound_no_finetune"] = ben_bound_no_finetune
    seed_results["time"] = time
    seed_results["posterior"] = model.get_post().detach().numpy()
    seed_results["KL"] = model.KL().item()
    if ben_bound_no_finetune != 2:
        seed_results["factor_no_finetune"] = ben_bound_no_finetune / final_bound['bound']
    else:
        seed_results["factor_no_finetune"] = 0
    return seed_results

def updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune, i):
    seed_results["seed"] = cfg.training.seed+i
    seed_results["train-error_finetune"] = train_error['error']
    seed_results["test-error_finetune"] = test_error['error']
    seed_results["ben_bound_with_finetune"] = ben_bound_with_finetune
    if ben_bound_with_finetune != 2:
        seed_results["factor_with_finetune"] = ben_bound_with_finetune / (seed_results["ben_bound_no_finetune"] / seed_results["factor_no_finetune"])
    else:
        seed_results["factor_with_finetune"] = 0
    return seed_results

def get_n_classes(dataset):
    if dataset in ["MUSH", "SVMGUIDE", "HABER", "TTT", "CODRNA", "ADULT", "PHIS"]:
        return 2
    elif dataset == "PROTEIN":
        return 3
    elif dataset == "SHUTTLE":
        return 7
    elif dataset in ["MNIST", "FASHION"]:
        return 10
    elif dataset == "SENSORLESS":
        return 11
    elif dataset == "PENDIGITS":
        return 12
    assert False, "Incorrect dataset"

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


def value_to_one_hot(values, n_classes):
    """
    Encodes a prediction array m x n into an m x n x n_classes one-hot matrix.
    """
    m = len(values)
    n = len(values[0])
    array = np.zeros((m, n, n_classes))
    for i in range(m):
        for j in range(n):
            array[i, j, int(values[i, j])] = 1
    return torch.tensor(array, dtype=torch.float)

def create_mu(theta, y_pred, i):
    w = theta.reshape(1, len(theta))
    y_pred_minus_y_i = y_pred - y_pred[:, :, [i]]
    mu_full = torch.matmul(w, torch.transpose(y_pred_minus_y_i, 0, 2)).squeeze().T
    correct_indexes = torch.arange(len(mu_full[0]))!=i
    return mu_full[:, correct_indexes], y_pred_minus_y_i[:, :, correct_indexes]

def create_Sigma(y_pred, i):
    y_pred_minus_y_i = y_pred - y_pred[:, :, [i]]
    Sigma_full = torch.matmul(torch.transpose(y_pred_minus_y_i, 1, 2), y_pred_minus_y_i)
    correct_indexes = torch.arange(len(Sigma_full[0])) != i
    return Sigma_full[:, correct_indexes][:, :, correct_indexes]

def custom_truncated_mean_multi_normal(x, mu, S):
    predictions = torch.distributions.multivariate_normal.MultivariateNormal(mu, S).sample(torch.Size([10000]))
    is_inf = torch.prod(predictions <= x, dim=1)
    predictions_lower = predictions[is_inf]
    return torch.mean(predictions_lower, dtype=torch.float, dim=0)

def create_notable_idx(unique_idx):
    len_unique_idx = len(unique_idx)
    notable_idx = []
    j = -1
    while True:
        j += 1
        for i in range(len_unique_idx):
            if unique_idx[i] == j:
                notable_idx.append(i)
                break
            elif j == len_unique_idx:
                return notable_idx

def purge_redundant_variables(y_pred_minus_y_i, mu, Sigma):
    y_preds_minus_y_i, mus, Sigmas = [], [], []
    for j in range(Sigma.shape[0]):
        diagonal_Sigma = torch.diag(Sigma[j])

        current_y_pred_minus_y_i = y_pred_minus_y_i[j, :, diagonal_Sigma > 0]
        current_mu = mu[j, diagonal_Sigma > 0]
        current_Sigma = Sigma[j, diagonal_Sigma > 0][:, diagonal_Sigma > 0]

        unique_idx = torch.unique(current_Sigma, dim=0, return_inverse=True)[1]
        notable_idx = create_notable_idx(unique_idx)

        y_preds_minus_y_i.append(current_y_pred_minus_y_i[:, notable_idx])
        mus.append(current_mu[notable_idx])
        Sigmas.append(current_Sigma[notable_idx][:, notable_idx])
    return y_preds_minus_y_i, mus, Sigmas

def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())

def multinomial_cdf_precomputations(y_pred, y_target, theta, n_classes, order):
    cdfs = []
    for i in range(n_classes):
        y_target_is_i = (y_target == i).squeeze()
        if torch.sum(y_target_is_i) > 0:
            one_hot_y_pred = value_to_one_hot(y_pred[y_target_is_i], n_classes)
            mu, y_pred_minus_y_i = create_mu(theta, one_hot_y_pred, i)
            Sigma = create_Sigma(one_hot_y_pred, i)
            purged_y_pred_minus_y_i, purged_mu, purged_Sigma = purge_redundant_variables(y_pred_minus_y_i, mu, Sigma)
            for j in range(len(mu)):
                cdfs.append(1 - MultinormalCDF.apply(purged_y_pred_minus_y_i[j], purged_mu[j], purged_Sigma[j]) ** order.item())
    return cdfs

class MultinormalCDF(torch.autograd.Function):
    """Cumulative distribution function of the multivariate normal distribution; its forward and backward passes"""

    @staticmethod
    def forward(ctx, one_hot_y_pred, purged_mu, purged_Sigma):
        ctx.save_for_backward(one_hot_y_pred, purged_mu, purged_Sigma)
        return torch.tensor(multivariate_normal.cdf(torch.zeros(len(purged_mu)), purged_mu, purged_Sigma, abseps=1e-2, releps=1e-2), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad):
        purged_y_pred_minus_y_i, purged_mu, purged_Sigma = ctx.saved_tensors
        return torch.tensor(0), -grad * torch.matmul(torch.inverse(purged_Sigma), torch.ones(len(purged_Sigma)) * 0.5), torch.tensor(0)