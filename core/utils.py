import math
import torch
import numpy as np
import random
import hydra

from torch.special import erf

from core.expected_risk import BetaInc
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

def whether_to_run_run(cfg):
    """
    Many tests ensuring that the current run has consistent hyperparameters.
    """
    assert cfg.training.distribution in ["categorical", "dirichlet", "gaussian"]
    if cfg.training.distribution == "categorical":
        assert cfg.model.prior == "adjusted"
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order
    elif cfg.training.distribution == "dirichlet":
        assert cfg.model.prior in ["adjusted", 1]
        assert cfg.training.risk != "Dis_Renyi"
    elif cfg.training.distribution == "gaussian":
        assert cfg.model.prior == 0
        if cfg.training.risk == "Dis_Renyi":
            assert 1 < cfg.bound.order < 2

    assert cfg.model.pred in ['UniformStumps', 'RandomForests', 'LinearClassifier'],  "Not a valid choice of model."
    if cfg.model.pred == 'LinearClassifier':
        assert cfg.model.output == 'embedding', "LinearClassifier implies embedding"
        assert cfg.dataset in ['CIFAR10_Inception_v3']
    elif cfg.model.pred == 'UniformStumps':
        assert cfg.model.output == 'class', "UniformStumps implies class"
        assert cfg.dataset in ['MUSH', 'TTT', 'HABER', 'PHIS', 'ADULT', 'CODRNA', 'SVMGUIDE']
    elif cfg.model.pred == 'RandomForests':
        assert cfg.model.output in ['class', 'proba'], "RandomForests implies class or proba"
        assert cfg.dataset in ['MNIST', 'PENDIGITS', 'PROTEIN', 'SENSORLESS', 'SHUTTLE', 'FASHION']

    assert cfg.training.risk in ['FO', 'SO', 'Bin', 'Dis_Renyi', 'Cbound', 'Test', 'VCdim']
    if cfg.training.risk == "Bin":
        assert cfg.training.rand_N > 0
    if cfg.training.risk == "Dis_Renyi":
        assert cfg.training.compute_disintegration, 'When using risk = Dis_Renyi, the disintegrated computation must be on.'


def create_root_dir(cfg):
    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.training.distribution}/"

    # Certain information are relevant to know only with some hyperparameters configurations.
    if cfg.model.pred == 'UniformStumps':
        ROOT_DIR += f"stmp-nt={cfg.model.stump_init}/"
    if cfg.model.pred == 'RandomForests':
        if cfg.training.distribution == 'gaussian':
            ROOT_DIR += f"output={cfg.model.output}/"

    if cfg.training.distribution == 'dirichlet':
        ROOT_DIR += f"prior={cfg.model.prior}/"

    if cfg.training.risk == 'Bin':
        ROOT_DIR += f"r-N={cfg.training.rand_N}/"
    if cfg.training.risk == 'Dis_Renyi':
        ROOT_DIR += f"order={cfg.bound.order}/"
    return ROOT_DIR


def initialize_predictors(cfg, data):
    if cfg.model.pred == "UniformStumps":
        return uniform_decision_stumps(cfg.model.n, data.X_train.shape[1], data.X_train.min(0),
                                       data.X_train.max(0), cfg.model.stump_init, cfg.training.distribution)
    elif cfg.model.pred == "RandomForests":
        if cfg.training.risk == "Test":
            m_train = int(len(data.X_train) * cfg.training.splits[0])
            return two_forests(cfg.model.n, data.X_train[:m_train], data.y_train[:m_train], samples_prop=cfg.model.samples_prop,
                               max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output, two_ways=False)
        elif cfg.training.risk == "VCdim":
            m_train = len(data.X_train) // 2
            return two_forests(cfg.model.n, data.X_train[:m_train], data.y_train[:m_train], samples_prop=cfg.model.samples_prop,
                               max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output, two_ways=False)
        return two_forests(cfg.model.n, data.X_train, data.y_train, samples_prop=cfg.model.samples_prop,
                           max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output, two_ways=True)
    elif cfg.model.pred == "LinearClassifier":
        # The linear classifier has its dataset being processed by a deep neural network implicitly.
        #   Therefore, no need for base classifiers computing predictions.
        return None, 1
    raise NotImplementedError


def updating_first_seed_results(seed_results, time, train_err, test_err, deterministic_bound, final_bound, part_bnd):
    # Some results are saved before the finetune (risk = FO) is done...
    seed_results["train-error"] = train_err['error']
    seed_results["test-error"] = test_err['error']
    seed_results["test-error_sampled"] = test_err['error_sampled']
    seed_results["test-error_sampled_std"] = test_err['error_sampled_std']
    seed_results["deterministic_bound"] = deterministic_bound
    seed_results["deterministic_bound_sampled"] = final_bound["bound_sampled"]
    seed_results["deterministic_bound_sampled_std"] = final_bound["bound_sampled_std"]
    seed_results["part_bnd"] = part_bnd
    seed_results["time"] = time
    return seed_results

def updating_last_seed_results(seed_results, cfg, train_error, test_error, part_bnd_tnd, i):
    # ... and other results after the finetune.
    seed_results["seed"] = cfg.training.seed+i
    seed_results["train-error_finetune"] = train_error['error']
    seed_results["test-error_finetune"] = test_error['error']
    seed_results["part_bnd_tnd"] = part_bnd_tnd
    return seed_results

def bin_cum(k, m, r):
    """
    Logarithm of P(x <= k), if X ~ Bin(m, r)
    """
    prob_cum = 0
    for i in range(k + 1):
        prob_cum += math.exp(log_prob_bin(torch.tensor(i), m, r))
    return prob_cum

def log_stirling_approximation(m):
    """
    Stirling's approximation for the logarithm of the factorial
    """
    if m == 0:
        return 0
    if m < 25:
        return math.log(math.factorial(m))
    return m * torch.log(m) - m + 0.5 * torch.log(2 * math.pi * m)


def log_binomial_coefficient(m, k):
    """
    Logarithm of the binomial coefficient using Stirling's approximation
    """
    return (log_stirling_approximation(m) -
            log_stirling_approximation(k) -
            log_stirling_approximation(m - k))

def log_prob_bin(k, m, r):
    """
    Logarithm of P(x = k), if X ~ Bin(m, r)
    """
    epsilon = torch.tensor(1e-10)
    if not torch.is_tensor(r):
        r = torch.tensor(r)
    return log_binomial_coefficient(m, k) + k * torch.log(torch.max(r, epsilon)) + \
                                      (m - k) * torch.log(torch.max(1 - r, epsilon))

def get_n_classes(dataset):
    """
    Given a dataset name, returns the number of classes it contains.
    """
    if dataset in ["MUSH", "SVMGUIDE", "HABER", "TTT", "CODRNA", "ADULT", "PHIS"]:
        return 2
    elif dataset == "PROTEIN":
        return 3
    elif dataset == "SHUTTLE":
        return 7
    elif dataset in ["CIFAR10_Inception_v3", "MNIST", "FASHION", "PENDIGITS"]:
        return 10
    elif dataset == "SENSORLESS":
        return 11
    elif dataset == "CIFAR100":
        return 100
    assert False, "Incorrect dataset"

def deterministic(random_state):
    """
    Set the random seed for all the packages that are used.
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def I(l, u):
    """
    Computes the incomplete beta function at x = 0.5.
    """
    return BetaInc.apply(l, u, torch.tensor(0.5), torch.tensor(1))

def Phi(z):
    """
    Computes the cumulative distribution function (CDF) of a standard unit gaussian function at point z.
    """
    return 1 / 2 * (1 - erf(z / 2 ** 0.5))