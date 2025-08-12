from torch import nn
import torch
from core.distributions import Gaussian


def pretrainedDNN(pred):
    if pred[:6] == 'resnet':
        model = torch.hub.load('pytorch/vision:v0.10.0', pred, weights=f"ResNet{pred[6:]}_Weights.DEFAULT")

        # We only keep the embedding
        model.fc = nn.Identity()
        model.eval()
    else:
        raise NotImplementedError("model.pred should be one the following: stumps-uniform, rf, resnet18, resnet34, resnet50, resnet101, resnet152.")
    return model

class LinearMultiClassifier(torch.nn.Linear):
    def __init__(self, input_size, output_size, bias, dtype, prior, distr="gaussian", kl_factor=1.):
        super().__init__(input_size, output_size, bias=bias, dtype=dtype)
        if distr != "gaussian":
            raise NotImplementedError

        post = torch.normal(0, 0.01, prior.shape, dtype=dtype)
        self.post = torch.nn.Parameter(post, requires_grad=True)

        self.prior = prior
        self.n_classes = output_size
        self.distribution = Gaussian(self.post, self.n_classes)
        self.distribution_name = distr
        self.kl_factor = kl_factor

    def forward(self, x):
        return x

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
        self.post = torch.nn.Parameter(value, requires_grad=True)
        self.distribution.w = self.post

    def random_draw_new_post(self):
        value = self.distribution.rsample()
        self.set_post(value.reshape(-1, self.n_classes))