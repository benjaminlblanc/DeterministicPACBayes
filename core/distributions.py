import torch
from torch.distributions.dirichlet import Dirichlet as Dir
from torch.distributions.multivariate_normal import MultivariateNormal as Gaus
import torch.nn.functional as F
from torch import lgamma, digamma
from core.utils import value_to_one_hot


def log_Beta(vec):
    return lgamma(vec).sum() - lgamma(vec.sum())

class Dirichlet():

    def __init__(self, alpha, a, n_classes, mc_draws=10):

        self.alpha = alpha
        self.mc_draws = mc_draws
        self.a = a
        self.n_classes = n_classes

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):

        exp_alpha = torch.exp(self.alpha)
        res = log_Beta(beta) - log_Beta(exp_alpha)
        res += torch.sum((exp_alpha - beta) * (digamma(exp_alpha) - digamma(exp_alpha.sum())))

        return res

    def Renyi(self, beta, order):

        exp_alpha = torch.exp(self.alpha)
        res = log_Beta(beta)
        res -= order / (order - 1) * log_Beta(exp_alpha)
        res += (1 / (order - 1)) * (log_Beta(order * exp_alpha + (1 - order) * beta) - log_Beta(exp_alpha))

        return res

    def KL_dis(self, beta):

        exp_alpha = torch.exp(self.alpha)
        res = Dir(exp_alpha).log_prob(exp_alpha / exp_alpha.sum())
        res -= Dir(beta).log_prob(beta / beta.sum())

        return res

    def risk(self, batch, mean=True, centered=None):
        alpha = torch.exp(self.alpha)
        y_target, y_pred = batch
        y_pred_oh = value_to_one_hot(y_pred, self.n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * alpha
        summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))
        if mean:
            return r.mean()
        return r.sum()

    def approximated_risk(self, batch, loss, mean=True):

        y_target, y_pred = batch

        thetas = torch.exp(self.alpha)

        r = loss(y_target, y_pred, thetas)

        if type(r) == tuple:
            if mean:
                    return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return sum(r) / len(y_target)

        return sum(r)


    def rsample(self):

        return Dir(torch.exp(self.alpha)).rsample()

    def mean(self):
        return Categorical(self.alpha, 0)

    def mode(self):
        assert all(self.alpha > 1), "can compute mode only of Dirichlet with alpha > 1"

        exp_alpha = torch.exp(self.alpha) - 1

        return exp_alpha / exp_alpha.sum()


class Gaussian():

    def __init__(self, w, a, n_classes, mc_draws=10):

        self.w = w
        self.a = a
        self.n_classes = n_classes
        self.mc_draws = mc_draws

    # Kullback-Leibler divergence between two Gaussian
    def KL(self, beta):
        return torch.sum((self.w - beta) ** 2 / 2)

    def Renyi(self, beta, order):
        return order * self.KL(beta)

    def KL_dis(self, beta):

        res = Gaus(self.w, torch.eye(len(self.w))).log_prob(self.w)
        res -= Gaus(beta, torch.eye(len(beta))).log_prob(beta)

        return res

    def risk(self, batch, mean=True, centered=None):
        y_target, y_pred = batch
        y_pred_oh = value_to_one_hot(y_pred, self.n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * self.w
        summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))

        if mean:
            return r.mean()
        return r.sum()


    def approximated_risk(self, batch, loss, mean=True):

        y_target, y_pred = batch

        thetas = self.w

        r = loss(y_target, y_pred, thetas)

        if type(r) == tuple:
            if mean:
                    return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return sum(r) / len(r)

        return sum(r)

    def rsample(self):

        return Gaus(self.w, torch.eye(len(self.w))).rsample()

    def mean(self):
        return Categorical(self.w, 0)

    def mode(self):
        return self.w


class Categorical():

    def __init__(self, theta, a, n_classes, mc_draws=10):
        self.theta = theta
        self.mc_draws = mc_draws
        self.a = a
        self.n_classes = n_classes
        
    def KL(self, beta):

        t = self.get_theta()

        b = beta / beta.sum()

        return (t * torch.log(t / b)).sum()

    def Renyi(self, beta, order):
        t = self.get_theta()

        b = beta / beta.sum()
        return (1 / (order - 1)) * torch.log((t ** order / b ** (order - 1)).sum())

    def KL_dis(self, beta):
        t = self.get_theta()
        assert (t ** 2).sum() == 1, 'To use KL_dis() on the Categorical distribution, the distribution must be centered on a one-hot vector.'

        b = beta / beta.sum()
        for i in range(len(t)):
            if t[i] == 1:
                return -torch.log(b[i])

    def approximated_risk(self, batch, loss, mean=True):

        t = self.get_theta()

        y_target, y_pred = batch

        r = loss(y_target, y_pred, t)

        if type(r) == tuple:
            if mean:
                return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return r.mean()
        return r.sum()

    def risk(self, batch, mean=True, centered=True):
        theta = self.get_theta() if centered else self.theta
        y_target, y_pred = batch
        y_pred_oh = value_to_one_hot(y_pred, self.n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * theta
        summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))

        if mean:
            return r.mean()
        return r.sum()

    def rsample(self):

        cum_pro = torch.cumsum(self.get_theta(), dim=0)
        cum_pro = torch.hstack((torch.zeros(1), cum_pro))
        value = torch.rand(1)
        for i in range(len(self.get_theta())):
            if cum_pro[i] <= value <= cum_pro[i+1]:
                return F.one_hot(torch.tensor(i), num_classes=len(self.get_theta())).to(torch.float64)

    def get_theta(self):
        return torch.nn.functional.softmax(self.theta, dim=0)


distr_dict = {
    "dirichlet": Dirichlet,
    "gaussian": Gaussian,
    "categorical": Categorical
}