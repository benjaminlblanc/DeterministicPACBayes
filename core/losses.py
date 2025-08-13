import torch
from core.utils import BetaInc, Phi, gaussian_cdf_precomputations, log_prob_bin, value_to_one_hot


def true_loss(y_target, y_pred, theta, distribution, n_classes):
    if distribution == "categorical":
        y_pred_oh = value_to_one_hot(y_pred, n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * theta
        summed_preds = torch.sum(weighted_preds, dim=2)
        return torch.nn.CrossEntropyLoss()(summed_preds, y_target)
    elif distribution == "dirichlet":
        pass
    elif distribution == "gaussian":
        if len(theta.shape) == 1:
            y_pred_oh = value_to_one_hot(y_pred, n_classes)
            weighted_preds = y_pred_oh.transpose(1, 2) * theta
            summed_preds = torch.sum(weighted_preds, dim=2)

        else:
            summed_preds = torch.matmul(y_pred, theta)
        return torch.nn.CrossEntropyLoss(reduction='none')(summed_preds, y_target.squeeze())


def triple_loss(y_target, y_pred, theta, predictor, distribution, n_classes, output_type):
    first_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1, output_type=output_type)
    if not torch.is_tensor(first_loss):
        first_loss = torch.tensor(first_loss)
    second_loss = torch.where(first_loss >= 0.5, first_loss, torch.zeros(1))
    third_loss = torch.where(first_loss < 0.5, first_loss, torch.zeros(1))
    if torch.sum(second_loss) == 0:
        if torch.sum(third_loss.nonzero()) == 0:
            return first_loss, torch.tensor(0.5, dtype=torch.float), torch.tensor(0, dtype=torch.float)
        return first_loss, torch.tensor(0.5, dtype=torch.float), third_loss[third_loss.nonzero()]
    elif torch.sum(third_loss) == 0:
        return first_loss, second_loss[second_loss.nonzero()], torch.tensor(0, dtype=torch.float)
    return first_loss, second_loss[second_loss.nonzero()], third_loss[third_loss.nonzero()]

def bin_loss(y_target, y_pred, theta, predictor, distribution, n_classes, n, output_type):
    first_order_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1, output_type=output_type)
    bin_loss = torch.zeros(len(first_order_loss))
    for i in range(n // 2, n + 1):
        bin_loss += torch.exp(log_prob_bin(torch.tensor(i), torch.tensor(n), first_order_loss))
    return bin_loss

def moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order, output_type):
    if predictor in ["rf", "stumps-uniform"]:
        if distribution == 'dirichlet':
            correct = torch.where(y_target == y_pred, theta, torch.zeros(1)).sum(1)
            wrong = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)
            return [BetaInc.apply(c, w, torch.tensor(0.5), torch.tensor(order)) for c, w in zip(correct, wrong)]

        elif distribution == 'gaussian':
            if n_classes == 2:
                inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(theta, (1, -1)) * y_pred, dim=1)) / torch.sum(y_pred ** 2, dim=1) ** 0.5
                return Phi(inner_Phi) ** order
            else:
                return gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, torch.tensor(order), predictor, output_type)

        elif distribution == 'categorical':
            return torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1) ** order
    elif distribution == 'gaussian':
        return gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, torch.tensor(order), predictor, output_type)