import math

import torch
from torch import lgamma, log1p, exp, log

from scipy.stats import multivariate_normal
from betaincder import betainc, betaincderp, betaincderq


def betaincderx(x, a, b):
    """
    Regularized incomplete beta derivative with respect to x.
    """
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    partial_x = exp((b - 1) * log1p(-x) + (a - 1) * log(x) - lbeta)
    return partial_x


class BetaInc(torch.autograd.Function):
    """
    Regularized incomplete beta function and its forward and backward passes, allowing auto-derivation.
    """
    @staticmethod
    def forward(ctx, p, q, x, order):

        x = torch.clamp(x, 0., 1.)

        ctx.save_for_backward(p, q, x, order)
        # deal with dirac distributions
        if p == 0.:
            return torch.tensor(1.)  # for any x, cumulative = 1.

        elif q == 0. or x == 0.:
            return torch.tensor(0.)  # cumulative = 0.

        return torch.tensor(betainc(x, p, q) ** order.item())

    @staticmethod
    def backward(ctx, grad):
        p, q, x, order = ctx.saved_tensors

        if p == 0. or q == 0. or x == 0.:  # deal with dirac distributions
            grad_p, grad_q, grad_x, grad_order = 0., 0., 0., 0.

        else:
            grad_p, grad_q, grad_x, grad_order = (betaincderp(x, p, q), betaincderq(x, p, q),
                                                  betaincderx(x, p, q), torch.zeros(1))

        return grad * grad_p, grad * grad_q, grad * grad_x, grad * grad_order


def create_mv_mu(theta, oh_y_pred_minus_oh_y_i):
    """
    Creation of the mu values for the majority vote gaussian stochastic average loss (see paper).
    """
    w = theta.reshape(1, len(theta))
    mu_full = torch.matmul(w, torch.transpose(oh_y_pred_minus_oh_y_i, 0, 2)).squeeze().T
    return mu_full


def create_mv_Sigma(oh_y_pred_minus_oh_y_i):
    """
    Creation of the Sigma values for the majority vote gaussian stochastic average loss (see paper).
    """
    Sigma_full = torch.matmul(torch.transpose(oh_y_pred_minus_oh_y_i, 1, 2), oh_y_pred_minus_oh_y_i)
    return Sigma_full


def create_nn_mu(theta, y_pred, i):
    """
    Creation of the mu values for the neural network gaussian stochastic average loss (see paper).
    """
    theta_minus_theta_i = theta - theta[:, [i]]
    mu_full = torch.matmul(y_pred, theta_minus_theta_i)
    correct_indexes = torch.arange(len(mu_full[0])) != i
    return mu_full[:, correct_indexes]


def create_nn_Sigma(y_pred):
    """
    Creation of the Sigma values for the neural network gaussian stochastic average loss (see paper).
    """
    return torch.sum(y_pred ** 2, dim=1)


def create_notable_idx(unique_idx):
    """
    Given a vector of indices, returns the first index for every unique index.
        Example: given unique_idx = tensor([0, 0, 0, 0, 0, 1, 1, 0, 0]), returns [0, 5].
    """
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
    """
    Given the means and the covariance matrix of a gaussian distribution, remove the redundant dimensions of the
        variable. Removes the dimensions for which that variance is 0. Only keep one dimensions out of several being
        perfectly correlated, if it happens.
    """
    y_preds_minus_y_i, mus, Sigmas = [], [], []
    for j in range(Sigma.shape[0]):
        diagonal_Sigma = torch.diag(Sigma[j])

        # Removes the dimensions for which that variance is 0.
        current_y_pred_minus_y_i = y_pred_minus_y_i[j, :, diagonal_Sigma > 0]
        current_mu = mu[j, diagonal_Sigma > 0]
        current_Sigma = Sigma[j, diagonal_Sigma > 0][:, diagonal_Sigma > 0]

        # Looks for perfectly correlated marginal distributions. Only keep one for every perfectly correlated sets.
        #   Since Sigma is symmetric, variables i and j are perfectly correlated only if Sigma[i] == Sigma[j].
        unique_idx = torch.unique(current_Sigma, dim=0, return_inverse=True)[1]
        notable_idx = create_notable_idx(unique_idx)

        y_preds_minus_y_i.append(current_y_pred_minus_y_i[:, notable_idx])
        mus.append(current_mu[notable_idx])
        Sigmas.append(current_Sigma[notable_idx][:, notable_idx])
    return y_preds_minus_y_i, mus, Sigmas


def gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, order, pred_type, output_type):
    """
    Prepare the data for computing the normal distribution CDF required by the gaussian average stochastic loss (see
        paper). For the majority vote, we compute the
    """
    cdfs = []
    mus = []
    Sigmas = []
    # Given the form of the average stochastic loss, we must iterate for each class of the labels and compute the loss
    #   for the examples in these classes independently.
    for i in range(n_classes):
        y_target_is_i = (y_target == i).squeeze()
        if torch.sum(y_target_is_i) > 0:
            if pred_type == "RandomForests":
                if output_type == 'class':
                    if n_classes == 2:
                        y_pred = (y_pred + 1) / 2
                        y_target = (y_target + 1) / 2
                    one_hot_y_pred = torch.nn.functional.one_hot(y_pred[y_target_is_i].to(torch.long),
                                                                 n_classes).to(torch.float)
                else:
                    one_hot_y_pred = y_pred[y_target_is_i]
                # Precomputation that serves in both the mu and the Sigma computation (once again, see paper).
                oh_y_pred_minus_oh_y_i = one_hot_y_pred - one_hot_y_pred[:, :, [i]]
                mu = create_mv_mu(theta, oh_y_pred_minus_oh_y_i)
                Sigma = create_mv_Sigma(oh_y_pred_minus_oh_y_i)
                # It is important not to keep the redundant dimensions of the resulting multivariate gaussian
                #   distribution; necessary for Sigma to be positive semi-definite.
                purged_y_pred_minus_y_i, purged_mu, purged_Sigma = purge_redundant_mv_variables(oh_y_pred_minus_oh_y_i,
                                                                                                mu, Sigma)
                # Each example yield a particular mu and Sigma; each of them must be treated independently.
                for j in range(len(mu)):
                    cdfs.append((1 - MultinormalCDF.apply(purged_mu[j], purged_Sigma[j])) ** order.item())
            elif pred_type == "LinearClassifier":
                # For the LinearClassifier, the resulting Sigma is such that we omit the covariance terms; the resulting
                #   multivariate gaussian has a CDF that can be computed as the product of each marginal CDF. This
                #   facilitates the computation and permits multiple computations (examples to be treated) at once.
                mus.append(create_nn_mu(theta, y_pred[y_target_is_i], i))
                Sigmas.append(create_nn_Sigma(y_pred[y_target_is_i]))
    if pred_type == "LinearClassifier":
        mu = torch.vstack(mus)
        Sigma = torch.hstack(Sigmas)
        cdfs += (1 - NormalCDF.apply(mu, Sigma)) ** order.item()
    return cdfs


def erf_approximation(x):
    """
    19th degree Taylor approximation of the erf function.
    """
    x = torch.clamp(x, -2, 2)
    summed = 0
    for i in range(20):
        summed += ((-1) ** i * x ** (2 * i + 1)) / (math.factorial(i) * (2 * i + 1))
    return summed * 2 / math.sqrt(math.pi)


class MultinormalCDF(torch.autograd.Function):
    """
    Cumulative distribution function of the multivariate normal distribution; its forward and backward passes.
    """
    @staticmethod
    def forward(ctx, purged_mu, purged_Sigma):
        ctx.save_for_backward(purged_mu, purged_Sigma)
        return torch.tensor(
            multivariate_normal.cdf(torch.zeros(len(purged_mu)), purged_mu, purged_Sigma, abseps=1e-2, releps=1e-2),
            dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad):
        purged_mu, purged_Sigma = ctx.saved_tensors
        # We estimate the gradient by omitting the covariance terms (see paper).
        truncated_expectation = -torch.sqrt(torch.diagonal(purged_Sigma) / (2 * math.pi)) * torch.exp(
                -1 / 2 * torch.matmul(torch.matmul(purged_mu.reshape(1, -1), torch.inverse(purged_Sigma)), purged_mu))
        return grad * torch.matmul(torch.inverse(purged_Sigma), truncated_expectation), torch.tensor(0)


class NormalCDF(torch.autograd.Function):
    """
    Cumulative distribution function of the normal distribution; its forward and backward passes.
    """
    @staticmethod
    def forward(ctx, mu, Sigma):
        ctx.save_for_backward(mu, Sigma)
        # We estimate the CDF of the multi gaussian distribution by the product of the marginal distributions.
        Sigma_ref = Sigma.unsqueeze(1).repeat(1, mu.shape[1])
        return torch.prod(1 / 2 * (1 - erf_approximation(mu / (torch.sqrt(2 * (2 * Sigma_ref))))), dim=1)

    @staticmethod
    def backward(ctx, grad):
        mu, Sigma = ctx.saved_tensors
        Sigma_repeated = Sigma.unsqueeze(1).repeat(1, mu.shape[1])
        # We estimate the gradient by omitting the covariance terms (see paper).
        truncated_expectation = -torch.sqrt((2 * Sigma_repeated) / (2 * math.pi)) * torch.exp(
            -1 / 2 * mu ** 2 / (2 * Sigma_repeated))
        return grad.unsqueeze(1).repeat(1, mu.shape[1]) * (
                    2 * Sigma_repeated) ** -1 * truncated_expectation, torch.tensor(0)
