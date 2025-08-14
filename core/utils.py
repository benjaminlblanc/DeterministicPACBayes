import math
import torch
import numpy as np
import random
import hydra

from torch.special import erf

from core.expected_risk import BetaInc
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

epsilon = torch.tensor(1e-10)

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

    assert cfg.training.risk in ['Tr', 'FO', 'SO', 'Bin', 'Dis_Renyi']
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
        return uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0),
                                       data.X_train.max(0), cfg.model.stump_init)
    elif cfg.model.pred == "RandomForests":
        return two_forests(cfg.model.M, data.X_train, data.y_train, samples_prop=cfg.model.samples_prop,
                           max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output)
    else:
        return None, 1


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
    elif dataset in ["CIFAR10_Inception_v3", "MNIST", "FASHION", "PENDIGITS"]:
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

def I(l, u):
    """
    Computes the incomplete beta function.
    """
    c = torch.tensor(0.5)
    return BetaInc.apply(l, u, c, torch.tensor(1))

def Phi(z):
    """
    Computes the Phi function.
    """
    return 1 / 2 * (1 - erf(z / 2 ** 0.5))