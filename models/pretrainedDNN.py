import torch
from core.distributions import Gaussian


class LinearMultiClassifier(torch.nn.Linear):
    """
    A variant of the majority vote, when the features are not predictions in themselves but representations.
    input_size (int): size of the representation.
    output_size (int): number of classes.
    prior (torch.tensor of size <number of base classifiers>): prior weights for every base classifier.
    kl_factor (float): KL penalty factor.
    """
    def __init__(self, input_size, output_size, prior, posterior_std, output_type):
        # For the sake of simplicity, we added a 1-valued variables, such that the bias is
        #   actually treated as a weight.
        super().__init__(input_size, output_size, bias=False)
        assert posterior_std > 0, "posterior_std must be bigger than 0."

        post = torch.normal(0, posterior_std, prior.shape)
        self.post = torch.nn.Parameter(post, requires_grad=True)

        self.prior = prior
        self.n_classes = output_size
        self.distribution = Gaussian(self.post, self.n_classes, output_type)

    def risk(self, batch, loss=None, mean=True):
        # If the loss function is given, then the deterministic risk is computed using this loss.
        #   Otherwise, the deterministic risk is computed.
        if loss is not None:
            return self.distribution.approximated_risk(batch, loss, mean)

        return self.distribution.deterministic_risk(batch, mean)

    def KL(self):
        return self.distribution.KL(self.prior)

    def Renyi(self, order):
        return self.distribution.Renyi(self.prior, order)

    def KL_dis(self):
        return self.distribution.KL_disintegrated(self.prior)

    def get_post(self):
        return self.post

    def get_unchanged_post(self):
        return self.post

    def get_post_grad(self):
        return self.post.grad

    def set_post(self, value):
        self.post = torch.nn.Parameter(value, requires_grad=True)
        self.distribution.w = self.post

    def random_draw_new_post(self):
        value = self.distribution.random_sample()
        self.set_post(value.reshape(-1, self.n_classes))