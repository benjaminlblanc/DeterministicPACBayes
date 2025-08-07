import torch
from core.utils import BetaInc, Phi, mv_gaussian_cdf_precomputations, log_prob_bin


def true_loss(y_target, y_pred, theta, distribution):
    input = torch.where(y_target != y_pred, theta, -theta).sum(1)
    heaviside = torch.heaviside(input, torch.tensor(0.))
    if distribution == 'categorical':
        return heaviside.detach() + input - input.detach()
    elif distribution in ['dirichlet', 'gaussian']:
        return [torch.sigmoid(inp) for inp in input]

def triple_loss(y_target, y_pred, theta, predictor, distribution, n_classes):
    first_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1)
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

def bin_loss(y_target, y_pred, theta, predictor, distribution, n_classes, n=100):
    first_order_loss = moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1)
    bin_loss = torch.zeros(len(first_order_loss))
    for i in range(n // 2, n + 1):
        bin_loss += torch.exp(log_prob_bin(torch.tensor(i), torch.tensor(n), first_order_loss))
    return bin_loss

def moment_loss(y_target, y_pred, theta, predictor, distribution, n_classes, order=1):
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
                return mv_gaussian_cdf_precomputations(y_pred, y_target, theta, n_classes, torch.tensor(order))

        elif distribution == 'categorical':
            return torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1) ** order
    elif predictor in ["resnet18"] and distribution == 'gaussian':
        return None