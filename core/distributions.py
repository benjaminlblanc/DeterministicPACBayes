import torch
from torch.distributions.dirichlet import Dirichlet as Dir
from torch.distributions.multivariate_normal import MultivariateNormal as Gaus
from torch import lgamma, digamma

from core.utils import BetaInc, Phi

class Dirichlet():

    def __init__(self, alpha, a, mc_draws=10):

        self.alpha = alpha
        self.mc_draws = mc_draws
        self.a = a

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        exp_alpha = torch.exp(self.alpha)
        res = lgamma(exp_alpha.sum()) - lgamma(exp_alpha).sum()
        res -= lgamma(beta.sum()) - lgamma(beta).sum()
        res += torch.sum((exp_alpha - beta) * (digamma(exp_alpha) - digamma(exp_alpha.sum())))

        return res

    def risk(self, batch, mean=True):
        # 01-loss applied to batch
        y_target, y_pred = batch
        exp_alpha = torch.exp(self.alpha)

        correct = torch.where(y_target == y_pred, exp_alpha, torch.zeros(1)).sum(1)
        wrong = torch.where(y_target != y_pred, exp_alpha, torch.zeros(1)).sum(1)
        
        s = [BetaInc.apply(c, w, torch.tensor(0.5 + self.a / 2)) for c, w in zip(correct, wrong)]

        if mean:
            return sum(s) / len(y_target)

        return sum(s)

    def approximated_risk(self, batch, loss, mean=True):

        y_target, y_pred = batch

        thetas = torch.exp(self.alpha) / torch.sum(torch.exp(self.alpha))

        r = loss(y_target, y_pred, thetas)

        if mean:
            return r.mean()

        return r.sum()


    def rsample(self):

        return Dir(torch.exp(self.alpha)).rsample((self.mc_draws,))

    def mean(self):
        return Categorical(self.alpha, 0)

    def mode(self):
        assert all(self.alpha > 1), "can compute mode only of Dirichlet with alpha > 1"

        exp_alpha = torch.exp(self.alpha) - 1

        return exp_alpha / exp_alpha.sum()

    def entropy(self, of_mean=True):

        if of_mean:
            return self.mean().entropy()
        else:
            return Dir(torch.exp(self.alpha)).entropy()


class Gaussian():

    def __init__(self, w, a, mc_draws=10):

        self.w = w
        self.mc_draws = mc_draws
        self.a = a

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):
        return torch.sum((self.w - beta) ** 2)

    def risk(self, batch, mean=True):
        # 01-loss applied to batch
        y_target, y_pred = batch
        inner_Phi = (torch.squeeze(y_target) * torch.sum(torch.reshape(self.w, (1, -1)) * y_pred, dim=1) - self.a) / torch.sum(y_pred ** 2, dim=1) ** 0.5

        s = Phi(inner_Phi)

        if mean:
            return sum(s) / len(y_target)

        return sum(s)

    def approximated_risk(self, batch, loss, mean=True):

        y_target, y_pred = batch

        thetas = self.w

        r = loss(y_target, y_pred, thetas)

        if mean:
            return r.mean()

        return r.sum()

    def rsample(self):

        return Gaus(self.w).rsample((self.mc_draws,))

    def mean(self):
        return Categorical(self.w, 0)

    def mode(self):
        return self.w

    def entropy(self, of_mean=True):

        if of_mean:
            return self.mean().entropy()
        else:
            return Gaus(self.w).entropy()


class Categorical():

    def __init__(self, theta, a, mc_draws=10):
        self.theta = theta
        self.mc_draws = mc_draws
        self.a = a
        
    def KL(self, beta):

        t = self.get_theta()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def approximated_risk(self, batch, loss, mean=True):

        t = self.get_theta()

        y_target, y_pred = batch

        r = loss(y_target, y_pred, t)
        
        if mean:
            return r.mean()

        return r.sum()

    def risk(self, batch, mean=True):

        t = self.get_theta()

        y_target, y_pred = batch

        w_theta = torch.where(y_target != y_pred, t, torch.zeros(1)).sum(1)

        r = (w_theta >= 0.5).float()

        if mean:
            return r.mean()

        return r.sum()

    def rsample(self):

        t = self.get_theta()

        return t.unsqueeze(0)

    def get_theta(self):
        
        return torch.nn.functional.softmax(self.theta, dim=0)

    def entropy(self):

        theta = self.get_theta()
        return - torch.sum(theta * torch.log(theta))

distr_dict = {
    "dirichlet": Dirichlet,
    "gaussian": Gaussian,
    "categorical": Categorical
}
