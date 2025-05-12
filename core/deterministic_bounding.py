import prtpy
import torch
import numpy as np
from core.utils import BetaInc, Phi
from tqdm import tqdm

def I(l, u, a):
    """
    Computes the incomplete beta function.
    """
    c = torch.tensor((1 + a) / 2)
    return BetaInc.apply(l, u, c, torch.tensor(1))

def deterministic_bound(Gibbs_risk, l, u, l_1_norm, leaf_values, distribution, a):
    """
    Computes Ben's bound, given Gibbs risk, u and l.
    """
    if distribution == "gaussian":
        biggest_norm = get_bound_on_pred_norm(leaf_values, max)
        smallest_norm = get_bound_on_pred_norm(leaf_values, min)
        phi_l, phi_u = Phi((l + a) / biggest_norm), Phi((u - a) / smallest_norm)
        return (Gibbs_risk - phi_u) / (1 - phi_l - phi_u)
    elif distribution == "dirichlet":
        I_l, I_u = I(l_1_norm - l, l, a), I(u, l_1_norm - u, a)
        return (Gibbs_risk - I_u) / (I_l - I_u)
    elif distribution == "categorical":
        return (Gibbs_risk - (1 - u)) / (l - (1 - u))

def get_indices(possible_values, sums):
    """
    Returns two vector of indices, associating the position of each element of possible_values
        in each vector sums[0] and sums[1].
    """
    tot, tot_1, tot_2 = [], [], []
    for i in range(len(sums[0])):
        cur = np.argwhere(sums[0][i] == possible_values)
        tot_1.append(cur[0][0])
        possible_values[cur[0][0]] = -1
    for i in range(len(sums[1])):
        cur = np.argwhere(sums[1][i] == possible_values)
        tot_2.append(cur[0][0])
        possible_values[cur[0][0]] = -1
    tot.append(tot_1)
    tot.append(tot_2)
    return tot

def get_normalized_l_u(leaf_values, normalized_tree_weights, leaf_type, distribution):
    """
    Returns the l and u values from the true-risk bound (see --).
    """
    remainder = 0
    if leaf_type == 'sign':
        if distribution == 'dirichlet':
            possible_values = torch.exp(normalized_tree_weights.clone().detach()).numpy()
        elif distribution == 'categorical':
            possible_values = torch.nn.functional.softmax(normalized_tree_weights, dim=0).clone().detach().numpy()
        elif distribution == 'gaussian':
            possible_values = normalized_tree_weights.clone().detach().numpy()
    else:
        normalized_leaf_values = np.reshape(normalized_tree_weights, (-1, 1)) * leaf_values
        possible_values = []
        for i in range(len(normalized_leaf_values)):
            possible_values.append((max(normalized_leaf_values[i])-min(normalized_leaf_values[i])) / 2)
            remainder += (max(normalized_leaf_values[i])+min(normalized_leaf_values[i])) / 2
        possible_values.append(remainder)
    sums = prtpy.partition(algorithm=prtpy.partitioning.ilp, numbins=2, items=np.sort(possible_values),
                           objective=prtpy.obj.MaximizeSmallestSum)
    indices = get_indices(possible_values.copy(), sums)
    if distribution == "gaussian":
        sum_1 = torch.sum(normalized_tree_weights[indices[0]])
        sum_2 = torch.sum(normalized_tree_weights[indices[1]])
        return torch.abs(sum_1 - sum_2), torch.sum(torch.abs(normalized_tree_weights)), None
    elif distribution == "categorical":
        sum_1 = torch.sum(torch.nn.functional.softmax(normalized_tree_weights, dim=0).clone().detach()[indices[0]])
        sum_2 = torch.sum(torch.nn.functional.softmax(normalized_tree_weights, dim=0).clone().detach()[indices[1]])
        biggest_sum = torch.max(sum_1, sum_2)
        smallest_sum = torch.min(sum_1, sum_2)
        return biggest_sum, biggest_sum + smallest_sum, None
    elif distribution == "dirichlet":
        sum_1 = torch.sum(torch.exp(normalized_tree_weights[indices[0]]))
        sum_2 = torch.sum(torch.exp(normalized_tree_weights[indices[1]]))
        biggest_sum = torch.max(sum_1, sum_2)
        smallest_sum = torch.min(sum_1, sum_2)
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

def compute_det_bound(model, bound, n, n_alphas, trainloader, loss, distribution_name, cur_PB_bound=None):
    """
    Pipeline for computing the deterministic bound.
    """
    leaves = np.ones((n_alphas, 2))
    leaves[:, 0] = -1
    l, u, l_1_norm = get_normalized_l_u(leaves, model.get_post(), 'sign', distribution_name)
    if cur_PB_bound is None:
        if type(trainloader) == tuple:
            train_data = trainloader[1], model(trainloader[0])
            cur_PB_bound = bound(n, model, model.risk(train_data, loss))
        else:
            cur_PB_bound = 0
            for _, batch in enumerate(trainloader):
                train_data = batch[1], model(batch[0])
                cur_PB_bound += (len(batch[1]) / n) * bound(n, model, model.risk(train_data, loss))
    return deterministic_bound(cur_PB_bound, l, u, l_1_norm, leaves, distribution_name, model.a)

def crop_weak_learners(model, n, bound, trainloader, loss, prior_coefficient, distribution_name):
    """
    Assigns small weights to predictors with medium weights, so that l and u might
        respectively be big and small.
    """
    best_alphas = model.get_post()
    sorted_alphas = torch.sort(torch.abs(best_alphas.clone()))[0]
    n_alphas = len(best_alphas)
    best_bound = compute_det_bound(model, bound, n, n_alphas, trainloader, loss, distribution_name)
    pbar = tqdm(range(12))
    low, up, strikes = 0, n_alphas-1, 0
    print(f"Current true-risk bound: {best_bound}.")
    for _ in pbar:
        if up - low <= 1:
            break
        mean = int((up + low) / 2)
        post = model.get_post().clone()
        post[torch.abs(best_alphas) <= sorted_alphas[mean]] = prior_coefficient
        model.set_post(post)
        cur_bound = compute_det_bound(model, bound, n, n_alphas, trainloader, loss, distribution_name)
        if cur_bound < best_bound:
            best_bound = cur_bound
            best_alphas = model.get_post()
            low = mean
            print(f"Current true-risk bound: {best_bound}.")
        else:
            post = best_alphas
            model.set_post(post)
            if strikes < 3:
                low = low + 2
            else:
                up = mean
    return model

def manual_model_finetune(model, n, bound, trainloader, loss, distribution_name):
    """
    Manually search for the best .
    """
    best_alphas = model.get_post()
    n_alphas = len(best_alphas)
    best_bound = compute_det_bound(model, bound, n, n_alphas, trainloader, loss, distribution_name)
    changed = True
    while changed:
        changed = False
        pbar = tqdm(range(len(model.post)))
        print(f"Current true-risk bound: {best_bound}.")
        for i in pbar:
            factor = torch.max(model.get_post()) / 100
            if best_alphas[i] >= factor:
                for change in ['min', 'max']:
                    # We slightly modify the current weighting
                    post = model.get_post()
                    post[i] = post[i] + (change == 'max') * factor - (change == 'min') * factor + 0.01 + torch.rand(1) * 0.001
                    model.set_post(post)
                    # And compute the resulting bound.
                    cur_bound = compute_det_bound(model, bound, n, n_alphas, trainloader, loss, distribution_name)
                    if cur_bound < best_bound:
                        best_bound = cur_bound
                        best_alphas = model.get_post()
                        changed = True
                    else:
                        post = model.get_post()
                        post[i] = best_alphas[i]
                        model.set_post(post)
    return model