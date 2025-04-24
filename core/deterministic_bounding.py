from scipy.special import erf
import prtpy
import torch
import numpy as np
from core.utils import BetaInc
from tqdm import tqdm


def Phi(z):
    """
    Computes the Phi function.
    """
    return 1 / 2 * (1 - erf(z / 2 ** 0.5))

def I(l, u, a):
    """
    Computes the incomplete beta function.
    """
    c = torch.tensor((1 + a) / 2)
    return BetaInc.apply(l, u, c)

def deterministic_bound(Gibbs_risk, l, u, l_1_norm, distribution, a):
    """
    Computes Ben's bound, given Gibbs risk, u and l.
    """
    if distribution == "gaussian":
        phi_l, phi_u = Phi(l), Phi(u)
        return (Gibbs_risk - phi_u) / (1 - phi_l - phi_u)
    elif distribution == "dirichlet":
        I_l, I_u = I(l_1_norm - l, l, a), I(u, l_1_norm - u, a)
        return (Gibbs_risk - I_u) / (I_l - I_u)

def get_normalized_l_u(leaf_values, normalized_tree_weights, leaf_type, distribution):
    """
    Returns the l and u values from the true-risk bound (see --).
    """
    remainder = 0
    if leaf_type == 'sign':
        possible_values = normalized_tree_weights
    else:
        normalized_leaf_values = np.reshape(normalized_tree_weights, (-1, 1)) * leaf_values
        possible_values = []
        for i in range(len(normalized_leaf_values)):
            possible_values.append((max(normalized_leaf_values[i])-min(normalized_leaf_values[i])) / 2)
            remainder += (max(normalized_leaf_values[i])+min(normalized_leaf_values[i])) / 2
        possible_values.append(remainder)
    sums = prtpy.partition(algorithm=prtpy.partitioning.ilp, numbins=2, items=np.sort(possible_values),
                           objective=prtpy.obj.MaximizeSmallestSum)
    biggest_sum = max(np.sum(sums[0]), np.sum(sums[1]))
    smallest_sum = min(np.sum(sums[0]), np.sum(sums[1]))
    if distribution == "gaussian":
        return ((biggest_sum - smallest_sum) / get_bound_on_pred_norm(leaf_values, max),
                np.sum(np.abs(possible_values)) - remainder), None
    elif distribution == "dirichlet":
        return biggest_sum, biggest_sum + smallest_sum, biggest_sum + smallest_sum

def get_bound_on_pred_norm(leaf_values, func):
    """
    Returns the biggest (or smallest) predictions norm: max_x (min_x) ||f(x)||.
    """
    leaf_values = leaf_values ** 2
    tot = 0
    for i in range(len(leaf_values)):
        tot += func(leaf_values[i])
    return tot ** 0.5

def compute_det_bound(model, bound, n, n_alphas, train_data, loss, cur_PB_bound=None):
    """
    Pipeline for computing the deterministic bound.
    """
    leaves = np.ones((n_alphas, 2))
    leaves[:, 0] = -1
    l, u, l_1_norm = get_normalized_l_u(leaves, model.get_post().detach().numpy(), 'sign', 'dirichlet')
    if cur_PB_bound is None:
        cur_PB_bound = bound(n, model, model.risk(train_data, loss))
    return deterministic_bound(cur_PB_bound, l, u, l_1_norm, 'dirichlet', 0).item()

def crop_weak_learners(model, n, bound, whole_batch, loss):
    """
    Assigns small weights to predictors with medium weights, so that l and u might
        respectively be big and small.
    """
    best_alphas = model.get_post()
    n_alphas = len(best_alphas)
    best_alphas[best_alphas < 1] = 1 / n_alphas
    # Making sure that every weight has a unique value.
    best_alphas += torch.rand(len(best_alphas)) * 0.001
    model.set_post(best_alphas)
    train_data = whole_batch[1], model(whole_batch[0])

    best_bound = compute_det_bound(model, bound, n, n_alphas, train_data, loss)
    changed = True
    while changed:
        changed = False
        print(f"Current true-risk bound: {best_bound}.")
        pbar = tqdm(range(len(model.post)))
        for i in pbar:
            if best_alphas[i] >= 1:
                for change in ['min', 'max']:
                    # We slightly modify the current weighting
                    post = model.get_post()
                    post[i] = post[i] + (change == 'max') * 1 - (change == 'min') * 1 + 0.01 + torch.rand(1) * 0.001
                    model.set_post(post)
                    # And compute the resulting bound.
                    cur_PB_bound = bound(n, model, model.risk(train_data, loss))
                    cur_bound = compute_det_bound(model, bound, n, n_alphas, train_data, loss, cur_PB_bound)
                    if cur_bound < best_bound and cur_bound > cur_PB_bound:
                        best_bound = cur_bound
                        best_alphas = model.get_post()
                        changed = True
                    else:
                        post = model.get_post()
                        post[i] = best_alphas[i]
                        model.set_post(post)
    return model