import torch
from torch.distributions.dirichlet import Dirichlet as Dir
from torch.distributions.multivariate_normal import MultivariateNormal as Gaus
import torch.nn.functional as F
from torch import lgamma, digamma


def log_Beta(vec):
    """
    Logarithm of the beta function.
    """
    return lgamma(vec).sum() - lgamma(vec.sum())


class Categorical:
    def __init__(self, theta, n_classes, _):
        # The theta parameters are real values. To access the true values parametrizing the distribution, we apply a
        #   softmax over these values (see self.get_theta).
        self.theta = theta
        self.n_classes = n_classes

    def KL(self, beta):
        # KL divergence between two categorical distributions
        t = self.get_post()
        return (t * torch.log(t / beta)).sum()

    def Renyi(self, beta, order):
        # Renyi divergence between two categorical distributions
        t = self.get_post()
        return (1 / (order - 1)) * torch.log((t ** order / beta ** (order - 1)).sum())

    def KL_disintegrated(self, beta):
        # KL divergence between two categorical distributions, with all the probability mass on a single point
        t = self.get_post()
        assert (t ** 2).sum() == 1, ('To use KL_dis() on the Categorical distribution, the distribution must be '
                                     'centered on a one-hot vector.')
        return (-t * torch.log(beta)).sum()

    def deterministic_risk(self, batch, mean=True):
        # Risk of the classifier centered on the mode of the distribution
        # No transformation is required if all the mass is on a given dimension
        centered = torch.prod(self.theta ** 2 == self.theta)
        theta = self.theta if centered else self.get_post()

        y_target, y_pred = batch
        if self.n_classes == 2:
            y_pred = (y_pred + 1) / 2
            y_target = (y_target + 1) / 2
        y_pred_oh = torch.nn.functional.one_hot(y_pred.to(torch.long), self.n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * theta
        summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))

        if mean:
            return r.mean()
        return r.sum()

    def approximated_risk(self, batch, loss, mean=True):
        # Risk of the average stochastic classifier, given a certain loss
        t = self.get_post()
        y_target, y_pred = batch
        r = loss(y_target, y_pred, t)

        if type(r) == tuple:
            # Triple loss
            if mean:
                return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return r.mean()
        return r.sum()

    def random_sample(self):
        # Random draw according to the distribution
        cum_pro = torch.cumsum(self.get_post(), dim=0)
        cum_pro = torch.hstack((torch.zeros(1), cum_pro))
        value = torch.rand(1)
        for i in range(len(self.get_post())):
            if cum_pro[i] <= value <= cum_pro[i + 1]:
                return F.one_hot(torch.tensor(i), num_classes=len(self.get_post())).to(torch.float64)

    def get_post(self):
        # Transformation to get the actual distribution parameters
        return torch.nn.functional.softmax(self.theta, dim=0)

    def get_unchanged_post(self):
        # Transformation to get the actual distribution parameters
        return self.theta

class Dirichlet:
    def __init__(self, alpha, n_classes, _):
        # The alpha parameters are real values. To access the true values parametrizing the distribution, we apply the
        #   exponential function over these values (see self.get_alpha).
        self.alpha = alpha
        self.n_classes = n_classes

    # Kullback-Leibler divergence between two Dirichlets
    def KL(self, beta):
        # KL divergence between two dirichlet distributions
        alphas = self.get_post()
        res = log_Beta(beta) - log_Beta(alphas)
        res += torch.sum((alphas - beta) * (digamma(alphas) - digamma(alphas.sum())))

        return res

    def Renyi(self, beta, order):
        # Renyi divergence between two dirichlet distributions
        alphas = self.get_post()
        res = log_Beta(beta)
        res -= log_Beta(alphas)
        res += (1 / (order - 1)) * (log_Beta(order * alphas + (1 - order) * beta) - log_Beta(alphas))

        return res

    def KL_disintegrated(self, beta):
        # KL divergence between two dirichlet distributions, with all the probability mass on a single point
        alphas = self.get_post()
        res = Dir(alphas).log_prob(alphas / alphas.sum())
        res -= Dir(beta).log_prob(beta / beta.sum())
        return res

    def deterministic_risk(self, batch, mean):
        centered = torch.prod(self.alpha > 0) and torch.sum(self.alpha) == 1
        alphas = self.alpha if centered else self.get_post()

        # Risk of the classifier centered on the mean of the distribution
        y_target, y_pred = batch
        if self.n_classes == 2:
            y_pred = (y_pred + 1) / 2
            y_target = (y_target + 1) / 2
        y_pred_oh = torch.nn.functional.one_hot(y_pred.to(torch.long), self.n_classes)
        weighted_preds = y_pred_oh.transpose(1, 2) * alphas
        summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))
        if mean:
            return r.mean()
        return r.sum()

    def approximated_risk(self, batch, loss, mean=True):
        # Risk of the average stochastic classifier, given a certain loss
        y_target, y_pred = batch
        alphas = self.get_post()
        r = loss(y_target, y_pred, alphas)

        if type(r) == tuple:
            # Triple loss
            if mean:
                    return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return sum(r) / len(y_target)
        return sum(r)

    def random_sample(self):
        # Random draw according to the distribution
        return Dir(torch.exp(self.alpha)).rsample()

    def get_post(self):
        # Transformation to get the actual distribution parameters
        return torch.exp(self.alpha)

    def get_unchanged_post(self):
        return self.alpha


class Gaussian:

    def __init__(self, w, n_classes, output_type):
        self.w = w
        self.n_classes = n_classes
        self.output_type = output_type

    def KL(self, beta):
        # KL divergence between two gaussian distributions
        return torch.sum((self.w - beta) ** 2 / 2)

    def Renyi(self, beta, order):
        # Renyi divergence between two gaussian distributions
        return order * self.KL(beta)

    def KL_disintegrated(self, beta):
        # KL divergence between two gaussian distributions, with all the probability mass on a single point
        res = Gaus(self.w, torch.eye(len(self.w))).log_prob(self.w)
        res -= Gaus(beta, torch.eye(len(beta))).log_prob(beta)
        return res

    def deterministic_risk(self, batch, mean=True):
        # Risk of the classifier centered on the mode of the distribution
        y_target, y_pred = batch
        if self.n_classes == 2:
            y_pred = (y_pred + 1) / 2
            y_target = (y_target + 1) / 2
        if self.output_type == 'embedding':
            summed_preds = torch.matmul(y_pred, self.w)
        else:
            y_pred_oh = torch.nn.functional.one_hot(y_pred.to(torch.long), self.n_classes)
            weighted_preds = y_pred_oh.transpose(1, 2) * self.w
            summed_preds = torch.sum(weighted_preds, dim=2)
        agg_preds = torch.argmax(summed_preds, dim=1).reshape(-1, 1)

        r = torch.where(y_target != agg_preds, torch.ones(1), torch.zeros(1))

        if mean:
            return r.mean()
        return r.sum()

    def approximated_risk(self, batch, loss, mean=True):
        # Risk of the average stochastic classifier, given a certain loss
        y_target, y_pred = batch
        r = loss(y_target, y_pred, self.w)

        if type(r) == tuple:
            # Triple loss
            if mean:
                    return torch.tensor([r[0].mean(), r[1].mean(), r[2].mean()])
            return (r[0].sum(), r[1].sum(), r[2].sum()), (len(r[0]), len(r[1]), len(r[2]))

        if mean:
            return sum(r) / len(r)
        return sum(r)

    def random_sample(self):
        # Random draw according to the distribution
        w = torch.reshape(self.w, (-1, 1))
        return Gaus(w.squeeze(), torch.eye(len(w))).rsample()

    def get_post(self):
        return self.w

    def get_unchanged_post(self):
        return self.w


distr_dict = {
    "categorical": Categorical,
    "dirichlet": Dirichlet,
    "gaussian": Gaussian
}