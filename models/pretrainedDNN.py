from torch import nn
import torch
from core.distributions import Gaussian


def pretrainedDNN(pred):
    if pred == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights="ResNet18_Weights.DEFAULT")

        # We only keep the embedding
        model.fc = nn.Identity()
        model.eval()
    else:
        raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf, resnet152]")
    return model

class LinearMultiClassifier(torch.nn.Linear):
    def __init__(self, input_size, output_size, bias, dtype, prior, distr="gaussian", kl_factor=1.):
        super().__init__(input_size, output_size, bias=bias, dtype=dtype)
        if distr != "gaussian":
            raise NotImplementedError

        self.num_voters = len(prior)
        post = torch.rand(self.num_voters) * 4 - 2  # uniform draws in [-2, 2]
        self.post = torch.nn.Parameter(post, requires_grad=True)

        self.prior = prior
        self.n_classes = output_size
        self.distribution = Gaussian(self.post, self.n_classes)
        self.distribution_name = distr
        self.kl_factor = kl_factor

    def forward(self, x):
        return x

    def risk(self, batch, loss=None, mean=True):

        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.risk(batch, mean)

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

        if self.distr_name == "categorical": # make sure params sum to 1
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
        if self.distribution_name == "dirichlet":
            value *= self.prior.sum()
        self.set_post(value)