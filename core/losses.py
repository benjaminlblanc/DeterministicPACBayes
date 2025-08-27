import torch

from core.expected_risk import gaussian_cdf_precomputations, BetaInc
from core.utils import Phi, log_prob_bin


def initialize_risk(cfg, n_classes):
    """
    Initialize the loss function, the bound coefficient, kl factor and divergence type given the considered risk.
    """
    dst = cfg.training.distribution
    if cfg.training.risk == "Tr":
        loss = lambda x, y, z: deterministic_loss(x, y, z, cfg.model.output)
        bound_coeff = 1.
        kl_factor = 1.
        div = 'KL'
    elif cfg.training.risk == "FO":
        loss =  lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, dst, n_classes, 1, cfg.model.output)
        bound_coeff = 1.
        kl_factor = 1.
        div = 'KL'
    elif cfg.training.risk == "SO":
        loss =  lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, dst, n_classes, 2, cfg.model.output)
        bound_coeff = 4.
        kl_factor = 2.
        div = 'KL'
    elif cfg.training.risk == "Bin":
        loss =  lambda x, y, z: bin_loss(x, y, z, cfg.model.pred, dst, n_classes, cfg.training.rand_n, cfg.model.output)
        bound_coeff = 2.
        kl_factor = cfg.training.rand_n
        div = 'KL'
    elif cfg.training.risk == "Dis_Renyi":
        loss = lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, dst, n_classes, 1, cfg.model.output)
        bound_coeff = 1.
        kl_factor = 1.
        div = 'Renyi'
    elif cfg.training.risk == "Cbound":
        loss = None
        bound_coeff = None
        kl_factor = None
        div = None
    else:
        raise NotImplementedError
    return loss, bound_coeff, kl_factor, div


def deterministic_loss(y_target, y_pred, theta, n_classes):
    """
    Compute the loss of the corresponding deterministic classifier.
    """
    if len(theta.shape) == 1:
        # Deterministic prediction for pred = StumpsUniform or RandomForest
        if n_classes == 2:
            y_pred = (y_pred + 1) / 2
            y_target = (y_target + 1) / 2
        y_pred_oh = torch.nn.functional.one_hot(y_pred.to(torch.long), n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * theta
        summed_preds = torch.sum(weighted_preds, dim=2)
        return torch.nn.CrossEntropyLoss(reduction='none')(summed_preds, y_target)
    else:
        # Deterministic prediction for pred = LinearClassifier
        summed_preds = torch.matmul(y_pred, theta)
        return torch.nn.CrossEntropyLoss(reduction='none')(summed_preds, y_target.squeeze())


def triple_loss(y_target, y_pred, theta, predictor, distribution, n_classes, output_type):
    """
    Computes the average loss, the average loss when an error is made, and the average loss when no error is made.
    """
    first_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1, output_type=output_type)
    if not torch.is_tensor(first_loss):
        first_loss = torch.tensor(first_loss)
    second_loss = torch.where(first_loss >= 0.5, first_loss, torch.zeros(1))
    third_loss = torch.where(first_loss < 0.5, first_loss, torch.zeros(1))
    if torch.sum(second_loss) == 0:
        # We take care of special cases, such as no error is made.
        if torch.sum(third_loss.nonzero()) == 0:
            return first_loss, torch.tensor(0.5, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        return first_loss, torch.tensor(0.5, dtype=torch.float), third_loss[third_loss.nonzero()]
    elif torch.sum(third_loss) == 0:
        return first_loss, second_loss[second_loss.nonzero()], torch.tensor(0, dtype=torch.float)
    return first_loss, second_loss[second_loss.nonzero()], third_loss[third_loss.nonzero()]

def bin_loss(y_target, y_pred, theta, predictor, distribution, n_classes, n, output_type):
    """
    Loss used for the computation of the binomial bound.
    """
    first_order_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1, output_type=output_type)
    bin_loss = torch.zeros(len(first_order_loss))
    for i in range(n // 2, n + 1):
        bin_loss += torch.exp(log_prob_bin(torch.tensor(i), torch.tensor(n), first_order_loss))
    return bin_loss

def moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order, output_type):
    """
    Moment loss (see the article). First moment (order = 1) corresponds to the "standard" stochastic loss.
    """
    if predictor in ["RandomForests", "UniformStumps"]:
        if distribution == 'categorical':
            return torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1) ** order
        elif distribution == 'dirichlet':
            correct = torch.where(y_target == y_pred, theta, torch.zeros(1)).sum(1)
            wrong = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)
            return [BetaInc.apply(c, w, torch.tensor(0.5), torch.tensor(order)) for c, w in zip(correct, wrong)]
        elif distribution == 'gaussian':
            if n_classes == 2:
                inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(theta,(1, -1)) * y_pred, dim=1)) /\
                            torch.sum(y_pred ** 2, dim=1) ** 0.5
                return Phi(inner_Phi) ** order
            else:
                return gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, torch.tensor(order),
                                                    predictor, output_type)
    elif predictor == "LinearClassifier":
        return gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, torch.tensor(order),
                                            predictor, output_type)