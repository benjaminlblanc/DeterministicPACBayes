import torch

from core.distributions import distr_dict

class MajorityVote(torch.nn.Module):

    def __init__(self, voters, prior, n_classes, distr="dirichlet", kl_factor=1.):

        super(MajorityVote, self).__init__()
        
        if distr not in ["dirichlet", "gaussian", "categorical"]:
            raise NotImplementedError

        self.num_voters = len(prior)
        if distr == "dirichlet":
            assert all(prior > 0), "all prior parameters must be positive"
            post = torch.rand(self.num_voters) * 2 + 1e-9  # uniform draws in (0, 2]
            self.post = torch.nn.Parameter(torch.log(post), requires_grad=True)  # use log (and apply exp(post) later so that posterior parameters are always positive)
        else:
            post = torch.rand(self.num_voters) * 4 - 2  # uniform draws in [-2, 2]
            self.post = torch.nn.Parameter(post, requires_grad=True)

        self.prior = prior
        self.voters = voters
        self.n_classes = n_classes
        self.distribution = distr_dict[distr](self.post, self.n_classes)
        self.distribution_name = distr
        self.kl_factor = kl_factor

    def forward(self, x):
        return x

    def voters_forward(self, x):
        return self.voters(x)

    def risk(self, batch, loss=None, mean=True, centered=True):

        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.risk(batch, mean, centered=centered)

    def KL(self):

        return self.kl_factor * self.distribution.KL(self.prior)

    def Renyi(self, order):

        return self.distribution.Renyi(self.prior, order)

    def KL_dis(self):
        return self.distribution.KL_dis(self.prior)

    def get_post(self):
        if self.distribution_name == "dirichlet":
            return torch.exp(self.post)
        return self.post * 1

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
        value = self.distribution.rsample()
        self.set_post(value)

class MultipleMajorityVote(torch.nn.Module):

    def __init__(self, voter_sets, priors, n_classes, weights, posteriors=None, distr="dirichlet",  kl_factor=1.):

        super(MultipleMajorityVote, self).__init__()

        assert len(voter_sets) == len(priors), "must specify same number of voter_sets and priors"
        assert sum(weights) == 1., weights

        if posteriors is not None:
            assert len(priors) == len(posteriors), "must specify same number of priors and posteriors"

            self.mvs = torch.nn.ModuleList([MajorityVote(voters, prior, n_classes, posterior=post, distr=distr, kl_factor=kl_factor) for voters, prior, post in zip(voter_sets, priors, posteriors)])

        else:
            self.mvs = torch.nn.ModuleList([MajorityVote(voters, prior, n_classes, distr=distr, kl_factor=kl_factor) for voters, prior in zip(voter_sets, priors)])

        self.weights = weights
        self.distribution_name = distr

    def forward(self, xs):

        return [mv.forward(x) for mv, x in zip(self.mvs, xs)]

    def voters_forward(self, xs):

        return [mv.voters_forward(x) for mv, x in zip(self.mvs, xs)]

    def risk(self, batchs, loss=None, mean=True, centered=True):

        risks = []
        for mv, w, batch in zip(self.mvs, self.weights, batchs):
            # import pdb; pdb.set_trace()
            risks.append(w * mv.risk(batch, loss, mean, centered))

        return sum(risks)

    def KL(self):

        return sum([w * mv.KL() for mv, w in zip(self.mvs, self.weights)])

    def Renyi(self, order):

        return sum([w * mv.Renyi(order) for mv, w in zip(self.mvs, self.weights)])

    def KL_dis(self):

        return sum([w * mv.KL_dis() for mv, w in zip(self.mvs, self.weights)])

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
