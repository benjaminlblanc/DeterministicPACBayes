import math
import torch
import numpy as np
import random

from scipy.stats import multivariate_normal
from betaincder import betainc, betaincderp, betaincderq
from torch import lgamma, log1p, exp, log
from torch.special import erf

epsilon = torch.tensor(1e-10)


def whether_to_run_run(cfg):
    if cfg.training.distribution in "dirichlet":
        assert cfg.model.prior in ["adjusted", 1]
        assert cfg.training.risk != "Dis_Renyi"
    elif cfg.training.distribution in "categorical":
        assert cfg.model.prior == "adjusted"
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order
    elif cfg.training.distribution == "gaussian":
        assert cfg.model.prior == 0
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order < 2
    if cfg.training.risk == "Bin":
        assert cfg.training.rand_n > 0

def updating_first_seed_results(seed_results, time, train_err, test_err, deterministic_bound, final_bound, ben_bound_no_finetune, triple_bound_no_finetune, ben_triple_bound_no_finetune):
    seed_results["train-error"] = train_err['error']
    seed_results["test-error"] = test_err['error']
    seed_results["test-error_sampled"] = test_err['error_sampled']
    seed_results["test-error_sampled_std"] = test_err['error_sampled_std']
    seed_results["deterministic_bound"] = deterministic_bound
    seed_results["deterministic_bound_sampled"] = final_bound["bound_sampled"]
    seed_results["deterministic_bound_sampled_std"] = final_bound["bound_sampled_std"]
    seed_results["ben_bound_no_finetune"] = ben_bound_no_finetune
    seed_results["triple_bound_no_finetune"] = triple_bound_no_finetune
    seed_results["ben_triple_bound_no_finetune"] = ben_triple_bound_no_finetune
    seed_results["time"] = time
    return seed_results

def updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune, triple_bound_with_finetune, ben_triple_bound_with_finetune, i):
    seed_results["seed"] = cfg.training.seed+i
    seed_results["train-error_finetune"] = train_error['error']
    seed_results["test-error_finetune"] = test_error['error']
    seed_results["ben_bound_with_finetune"] = ben_bound_with_finetune
    seed_results["triple_bound_with_finetune"] = triple_bound_with_finetune
    seed_results["ben_triple_bound_with_finetune"] = ben_triple_bound_with_finetune
    return seed_results

def log_stirling_approximation(n):
    """
    Stirling's approximation for the logarithm of the factorial
    """
    if n == 0:
        return 0
    return n * torch.log(n) - n + 0.5 * torch.log(2 * math.pi * n)


def log_binomial_coefficient(n, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation
    """
    return (log_stirling_approximation(n) -
            log_stirling_approximation(k) -
            log_stirling_approximation(n - k))

def log_prob_bin(k, n, r):
    """
    Logarithm of P(x = k), if X ~ Bin(n, r)
    """
    return log_binomial_coefficient(n, k) + k * torch.log(torch.max(r, epsilon)) + (n - k) * torch.log(torch.max(1 - r, epsilon))

def find_ns(risks, n):
    if risks[1] == risks[2]:
        return n, n // 2, n // 2
    elif risks[1] == 0:
        return n, 1, n-1
    elif risks[2] == 0.5:
        return n, n-1, 1
    p = (risks[0] - risks[2]) / (risks[1] - risks[2])
    return n, max(int(p * n), 1), max(int((1-p) * n), 1)

def get_n_classes(dataset):
    if dataset in ["MUSH", "SVMGUIDE", "HABER", "TTT", "CODRNA", "ADULT", "PHIS"]:
        return 2
    elif dataset == "PROTEIN":
        return 3
    elif dataset == "SHUTTLE":
        return 7
    elif dataset in ["CIFAR10", "MNIST", "FASHION", "PENDIGITS"]:
        return 10
    elif dataset == "SENSORLESS":
        return 11
    elif dataset == "CIFAR100":
        return 100
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

def create_mv_mu(theta, oh_y_pred_minus_oh_y_i):
    w = theta.reshape(1, len(theta))
    mu_full = torch.matmul(w, torch.transpose(oh_y_pred_minus_oh_y_i, 0, 2)).squeeze().T
    return mu_full

def create_mv_Sigma(oh_y_pred_minus_oh_y_i):
    Sigma_full = torch.matmul(torch.transpose(oh_y_pred_minus_oh_y_i, 1, 2), oh_y_pred_minus_oh_y_i)
    return Sigma_full

def create_nn_mu(theta, y_pred, i):
    theta_minus_theta_i = theta - theta[:, [i]]
    mu_full = torch.matmul(y_pred, theta_minus_theta_i)
    correct_indexes = torch.arange(len(mu_full[0])) != i
    return mu_full[:, correct_indexes]

def create_nn_Sigma(y_pred):
    return torch.sum(y_pred ** 2, dim=1)

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

def purge_redundant_mv_variables(y_pred_minus_y_i, mu, Sigma):
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

def gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, order, pred_type):
    cdfs = []
    mus = []
    Sigmas = []
    for i in range(n_classes):
        y_target_is_i = (y_target == i).squeeze()
        if torch.sum(y_target_is_i) > 0:
            if pred_type == "rf":
                one_hot_y_pred = value_to_one_hot(y_pred[y_target_is_i], n_classes)
                oh_y_pred_minus_oh_y_i = one_hot_y_pred - one_hot_y_pred[:, :, [i]]
                mu = create_mv_mu(theta, oh_y_pred_minus_oh_y_i)
                Sigma = create_mv_Sigma(oh_y_pred_minus_oh_y_i)
                purged_y_pred_minus_y_i, purged_mu, purged_Sigma = purge_redundant_mv_variables(oh_y_pred_minus_oh_y_i, mu, Sigma)
                for j in range(len(mu)):
                    cdfs.append(1 - MultinormalCDF.apply(purged_mu[j], purged_Sigma[j]) ** order.item())
            else:
                mus.append(create_nn_mu(theta, y_pred[y_target_is_i], i))
                Sigmas.append(create_nn_Sigma(y_pred[y_target_is_i]))
    mu = torch.vstack(mus)
    Sigma = torch.hstack(Sigmas)
    cdfs += 1 - NormalCDF.apply(mu, Sigma) ** order.item()
    return cdfs

class MultinormalCDF(torch.autograd.Function):
    """Cumulative distribution function of the multivariate normal distribution; its forward and backward passes"""

    @staticmethod
    def forward(ctx, purged_mu, purged_Sigma):
        ctx.save_for_backward(purged_mu, purged_Sigma)
        return torch.tensor(multivariate_normal.cdf(torch.zeros(len(purged_mu)), purged_mu, purged_Sigma, abseps=1e-2, releps=1e-2), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad):
        purged_mu, purged_Sigma = ctx.saved_tensors
        truncated_expectation = -torch.sqrt(torch.diagonal(purged_Sigma) / (2 * math.pi)) * torch.exp(-1/2 * purged_mu ** 2 / purged_Sigma)
        return grad * torch.matmul(torch.inverse(purged_Sigma), truncated_expectation), torch.tensor(0)

class NormalCDF(torch.autograd.Function):
    """Cumulative distribution function of the normal distribution; its forward and backward passes"""

    @staticmethod
    def forward(ctx, mu, Sigma):
        ctx.save_for_backward(mu, Sigma)
        return torch.prod(1/2 * (1 - erf_approximation(mu / (torch.sqrt(2 * (2 * Sigma.unsqueeze(1).repeat(1, mu.shape[1])))))), dim=1).to(torch.double)

    @staticmethod
    def backward(ctx, grad):
        mu, Sigma = ctx.saved_tensors
        Sigma_repeated = Sigma.unsqueeze(1).repeat(1, mu.shape[1])
        truncated_expectation = -torch.sqrt((2 * Sigma_repeated) / (2 * math.pi)) * torch.exp(-1/2 * mu ** 2 / (2 * Sigma_repeated))
        return grad.unsqueeze(1).repeat(1, mu.shape[1]) * (2 * Sigma_repeated) ** -1 * truncated_expectation, torch.tensor(0)


def erf_approximation(x):
    x = torch.clamp(x, -2, 2)
    summed = 0
    for i in range(20):
        summed += ((-1) ** i * x ** (2 * i + 1)) / (math.factorial(i) * (2 * i + 1))
    return summed * 2 / math.sqrt(math.pi)