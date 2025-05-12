import torch

from core.utils import BetaInc, Phi


def bin_loss(y_target, y_pred, theta, distribution, n=100):

    if distribution == 'dirichlet':
        correct = torch.where(y_target == y_pred, theta, torch.zeros(1)).sum(1)
        wrong = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)
        w_theta = [BetaInc.apply(c, w, torch.tensor(0.5), torch.tensor(1)) for c, w in zip(correct, wrong)]

    elif distribution == 'gaussian':
        inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(theta, (1, -1)) * y_pred, dim=1)) / torch.sum(y_pred ** 2, dim=1) ** 0.5
        w_theta = Phi(inner_Phi)

    elif distribution == 'categorical':
        w_theta = torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1)

    return torch.stack([BetaInc.apply(torch.tensor(n // 2 + 1), torch.tensor(n // 2), w, torch.tensor(1)) for w in w_theta])

def moment_loss(y_target, y_pred, theta, distribution, order=1):

    assert order in [1, 2], "only first and second order supported atm"

    if distribution == 'dirichlet':
        correct = torch.where(y_target == y_pred, theta, torch.zeros(1)).sum(1)
        wrong = torch.where(y_target != y_pred, theta, torch.zeros(1)).sum(1)
        return [BetaInc.apply(c, w, torch.tensor(0.5), torch.tensor(order)) for c, w in zip(correct, wrong)]

    elif distribution == 'gaussian':
        inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(theta, (1, -1)) * y_pred, dim=1)) / torch.sum(y_pred ** 2, dim=1) ** 0.5
        return Phi(inner_Phi) ** order

    elif distribution == 'categorical':
        return torch.where(y_target != y_pred, theta, torch.tensor(0.)).sum(1) ** order