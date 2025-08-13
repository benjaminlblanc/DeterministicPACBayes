import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):
    """
    The basic class for the majority vote.
    prior (torch.tensor of size <number of base classifiers>): prior weights for every base classifier.
    n_classes (int): number of classes for the given task.
    kl_factor (float): KL penalty factor.
    output_type (str): type of prediction rendered by the base classifiers.
    """
    def __init__(self, voters, prior, n_classes, distr_name, kl_factor=1., output_type='class'):
        super(MajorityVote, self).__init__()
        
        if distr_name not in ["dirichlet", "gaussian", "categorical"]:
            raise NotImplementedError

        self.num_voters = len(prior)
        self.distribution_name = distr_name
        self.n_classes = n_classes
        self.output_type = output_type

        # We inialize the current distribution.
        self.post = None
        self.distribution = distr_dict[distr_name](self.post, self.n_classes, self.output_type)

        # We initialize the distribution initial posterior, depending on the distribution name.
        if distr_name == "dirichlet":
            post = torch.rand(self.num_voters) * 2 + 1e-9  # uniform draws in (0, 2]
        else:
            post = torch.rand(self.num_voters) * 4 - 2  # uniform draws in [-2, 2]
        self.set_post(post)

        self.prior = prior
        self.voters = voters
        self.kl_factor = kl_factor

    def voters_forward(self, x):
        return self.voters(x)

    def risk(self, batch, loss=None, mean=True):
        # If the loss function is given, then the deterministic risk is computed using this loss.
        #   Otherwise, the deterministic risk is computed.
        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.deterministic_risk(batch, mean)

    def KL(self):
        return self.kl_factor * self.distribution.KL(self.prior)

    def Renyi(self, order):
        return self.distribution.Renyi(self.prior, order)

    def KL_disintegrated(self):
        return self.distribution.KL_disintegrated(self.prior)

    def get_post(self):
        if self.distribution_name == "dirichlet":
            return torch.exp(self.post)
        return self.post

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):

        assert len(value) == self.num_voters
         
        if self.distribution_name == "categorical": # make sure params sum to 1
            self.post = torch.nn.Parameter(value, requires_grad=True)
            self.distribution.theta = self.post

        elif self.distribution_name == "dirichlet":
            assert all(value > 0), "all posterior parameters must be positive"
            self.post = torch.nn.Parameter(torch.log(value), requires_grad=True) # use log (and apply exp(post) later so that posterior parameters are always positive)
            self.distribution.alpha = self.post

        elif self.distribution_name == "gaussian":
            self.post = torch.nn.Parameter(value, requires_grad=True)
            self.distribution.w = self.post

    def random_draw_new_post(self):
        value = self.distribution.random_sample()
        self.set_post(value)

class MultipleMajorityVote(torch.nn.Module):
    """
    Create a set of majority votes, given a set of random forest classifiers.
    prior (torch.tensor of size <number of base classifiers>): prior weights for every base classifier.
    n_classes (int): number of classes for the given task.
    weights (tuple of floats summing to 1): weight of each base majority vote.
    kl_factor (float): KL penalty factor.
    output_type (str): type of prediction rendered by the base classifiers.
    """
    def __init__(self, voter_sets, priors, n_classes, weights, distr_name, kl_factor=1., output_type='class'):

        super(MultipleMajorityVote, self).__init__()

        assert len(voter_sets) == len(priors), "must specify same number of voter_sets and priors"
        assert sum(weights) == 1., weights

        self.mvs = torch.nn.ModuleList([MajorityVote(voters, prior, n_classes, distr_name, kl_factor, output_type)
                                        for voters, prior in zip(voter_sets, priors)])
        self.weights = weights
        self.distribution_name = distr_name

    def voters_forward(self, xs):
        return [mv.voters_forward(x) for mv, x in zip(self.mvs, xs)]

    def risk(self, batchs, loss=None, mean=True):
        risks = []
        for mv, w, batch in zip(self.mvs, self.weights, batchs):
            risks.append(w * mv.risk(batch, loss, mean))
        return sum(risks)

    def KL(self):
        return sum([w * mv.KL() for mv, w in zip(self.mvs, self.weights)])

    def Renyi(self, order):
        return sum([w * mv.Renyi(order) for mv, w in zip(self.mvs, self.weights)])

    def KL_disintegrated(self):
        return sum([w * mv.KL_disintegrated() for mv, w in zip(self.mvs, self.weights)])

    def get_post(self):
        return torch.cat([mv.get_post() for mv in self.mvs], 0)

    def get_post_grad(self):
        return torch.cat([mv.post.grad for mv in self.mvs], 0)

    def set_post(self, value):
        value = torch.reshape(value, (len(self.mvs), -1))
        for i in range(len(self.mvs)):
            self.mvs[i].set_post(value[i])

    def random_draw_new_post(self):
        for i in range(len(self.mvs)):
            self.mvs[i].random_draw_new_post()
