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

    assert cfg.training.risk in ['Tr', 'FO', 'SO', 'Bin', 'Dis_Renyi', 'Cbound']
    if cfg.training.risk == "Tr":
        assert cfg.bound.type == "triple"
    if cfg.training.risk == "Bin":
        assert cfg.training.rand_n > 0
    if cfg.training.risk == "Dis_Renyi":
        assert cfg.training.compute_disintegration, 'When using risk = Dis_Renyi, the disintegrated computation mus tbe on.'


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
        ROOT_DIR += f"r-n={cfg.training.rand_n}/"
    if cfg.training.risk == 'Dis_Renyi':
        ROOT_DIR += f"order={cfg.bound.order}/"
    return ROOT_DIR


def initialize_predictors(cfg, data):
    if cfg.model.pred == "UniformStumps":
        return uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0),
                                       cfg.model.stump_init, cfg.training.distribution)
    elif cfg.model.pred == "RandomForests":
        return two_forests(cfg.model.M, data.X_train, data.y_train, samples_prop=cfg.model.samples_prop,
                           max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output)
    elif cfg.model.pred == "LinearClassifier":
        # The linear classifier has its dataset being processed by a deep neural network implicitely.
        #   Therefore, no need for base classifiers computing predictions.
        return None, 1
    raise NotImplementedError


def updating_first_seed_results(seed_results, time, train_err, test_err, deterministic_bound, final_bound, part_bnd, triple_bnd, part_triple_bnd):
    # Some results are saved before the finetune (risk = FO) is done...
    seed_results["train-error"] = train_err['error']
    seed_results["test-error"] = test_err['error']
    seed_results["test-error_sampled"] = test_err['error_sampled']
    seed_results["test-error_sampled_std"] = test_err['error_sampled_std']
    seed_results["deterministic_bound"] = deterministic_bound
    seed_results["deterministic_bound_sampled"] = final_bound["bound_sampled"]
    seed_results["deterministic_bound_sampled_std"] = final_bound["bound_sampled_std"]
    seed_results["part_bnd"] = part_bnd
    seed_results["triple_bnd"] = triple_bnd
    seed_results["part_triple_bnd"] = part_triple_bnd
    seed_results["time"] = time
    return seed_results

def updating_last_seed_results(seed_results, cfg, train_error, test_error, part_bnd_tnd, triple_bnd_tnd, part_triple_bnd_tnd, i):
    # ... and other results after the finetune.
    seed_results["seed"] = cfg.training.seed+i
    seed_results["train-error_finetune"] = train_error['error']
    seed_results["test-error_finetune"] = test_error['error']
    seed_results["part_bnd_tnd"] = part_bnd_tnd
    seed_results["triple_bnd_tnd"] = triple_bnd_tnd
    seed_results["part_triple_bnd_tnd"] = part_triple_bnd_tnd
    return seed_results

def log_stirling_approximation(n):
    """
    Stirling's approximation for the logarithm of the factorial
    """
    if n == 0:
        return 0
    if n < 25:
        return math.log(math.factorial(n))
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
    epsilon = torch.tensor(1e-10)
    return log_binomial_coefficient(n, k) + k * torch.log(torch.max(r, epsilon)) + \
                                      (n - k) * torch.log(torch.max(1 - r, epsilon))

def find_ns(risks, n):
    """
    Given two a vectors avg risks = R/n, R_1/n_1, R_2/n_2, where n = n_1 + n_2 and R = R_1 + R_2, and n,
        we find and return n, n_1, n_2.
    """
    if risks[1] == risks[2]:
        return n, n // 2, n // 2
    elif risks[1] == 0:
        # These values are used for computing PAC-Bayes bounds, so n_1=0 or n_2=0 would lead to errors.
        return n, 1, n - 1
    elif risks[2] == 0.5:
        # These values are used for computing PAC-Bayes bounds, so n_1=0 or n_2=0 would lead to errors.
        return n, n - 1, 1
    p = (risks[0] - risks[2]) / (risks[1] - risks[2])
    return n, max(int(p * n), 1), max(int((1-p) * n), 1)

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