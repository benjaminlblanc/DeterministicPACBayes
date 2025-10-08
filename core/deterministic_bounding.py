import prtpy
import torch
import numpy as np
from torch.utils.data import DataLoader

from core.utils import Phi, I
from tqdm import tqdm


def get_indices(possible_values, sums):
    """
    Returns two vector of indices, associating the position of each element of possible_values
        in each vector sums[0] and sums[1].
    Example: possible_values = [0, 2, 6, 1] and sums = [[1, 2], [0, 6]], returns [[3, 1], [0, 2]].
    """
    tot_1, tot_2 = [], []
    for i in range(len(sums[0])):
        cur = np.argwhere(sums[0][i] == possible_values)
        tot_1.append(cur[0][0])
        possible_values[cur[0][0]] = -1
    for i in range(len(sums[1])):
        cur = np.argwhere(sums[1][i] == possible_values)
        tot_2.append(cur[0][0])
        possible_values[cur[0][0]] = -1
    return [tot_1, tot_2]

def get_b_c(possible_values, n, distribution, multiclass=False):
    """
    Returns the l and u values from the true-risk bound (see --).
    """
    possible_values_np = possible_values.detach().numpy()
    # The partition bound is not valid for the multiclass gaussian approach
    if distribution == "gaussian" and multiclass:
        return torch.tensor(0), torch.tensor(0.5)

    try:
        # This function computes the partitioning problem (2 bins)
        sums = prtpy.partition(algorithm=prtpy.partitioning.ilp,
                               items=np.sort(possible_values_np),
                               objective=prtpy.obj.MaximizeSmallestSum,
                               numbins=2)
    except ValueError:
        # In cases where the function do not converge to a solution, we avoid a crash by doing as follows
        sums = [[1], [1]]
        possible_values_np = np.array([1, 1])
        possible_values = torch.tensor([1, 1])
    # We gather the indices of the values in each bins given by the partitioning algorithm's solution
    indices = get_indices(possible_values_np.copy(), sums)

    sum_1 = torch.sum(possible_values[indices[0]])
    sum_2 = torch.sum(possible_values[indices[1]])
    biggest_sum = torch.max(sum_1, sum_2)
    smallest_sum = torch.min(sum_1, sum_2)
    if distribution == "categorical":
        return 0, biggest_sum
    elif distribution == "dirichlet":
        return I(biggest_sum + smallest_sum, torch.tensor(0)), I(smallest_sum, biggest_sum)
    elif distribution == "gaussian":
        return Phi(torch.sum(torch.abs(possible_values)) / n ** 0.5), 1 - Phi((biggest_sum - smallest_sum) / n ** 0.5)

def compute_bound(model, bound, n, trainloader, loss, disintegrated):
    """
    Pipeline for computing a bound.
    """
    if type(trainloader) == DataLoader: # pred in [StumpsUniform, LinearClassifier]
        cur_PB_bound = 0
        for _, batch in enumerate(trainloader):
            train_data = batch[1], batch[0]
            cur_PB_bound += (len(batch[1]) / n) * bound(n, model, model.risk(train_data, loss), disintegrated)
    elif type(trainloader) == list: # pred == RandomForests
        cur_PB_bound, count = 0, 0
        pbar = range(len(trainloader[0]))
        for i, *batches in zip(pbar, *trainloader):
            X = [batch[0] for batch in batches]
            n = sum(map(len, X))
            data = [(batches[i][1], X[i]) for i in range(len(batches))]
            cur_PB_bound += len(data[0][0]) * bound(n, model, model.risk(data, loss), disintegrated)
            count += len(data[0][0])
        cur_PB_bound /= count
    else:
        cur_PB_bound = bound(n, model, model.risk(trainloader, loss), disintegrated)
    return cur_PB_bound

def compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name, Gibbs_risk=None, multiclass=False):
    """
    Pipeline for computing the deterministic bound.
    """
    post = model.get_post()
    b, c = get_b_c(post, n, distribution_name, multiclass)

    # If the bound on the Gibbs risk is not given, we have to compute it
    if Gibbs_risk is None:
        Gibbs_risk = compute_bound(model, bound, m, trainloader, loss, False)

    return (Gibbs_risk - b) / (c - b)

def clip_weak_learners(model, m, bound, trainloader, loss, prior_coefficient, distribution_name):
    """
    Assigns small weights to predictors with medium weights, so b and c are the smallest possible (partition bound).
    """
    best_post = model.get_unchanged_post().clone()
    sorted_post = torch.sort(torch.abs(best_post.clone()))[0]
    post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
    print(f"\nCurrent partition bound: {round(best_bound.item(), 4)}.")
    print("Clipping weak learners...")
    pbar = tqdm(list(range(10, 100, 10)) + list(range(91, 100, 1)))
    for i in pbar:
        max_idx = int(n * i / 100) - 1
        # We try several simplification of the posterior by clipping the smallest values to the prior values. Having
        #   lesser small values help to obtain better partitioning bound.
        post[torch.abs(best_post) <= sorted_post[max_idx]] = prior_coefficient
        model.set_post(post)
        cur_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
        if cur_bound < best_bound:
            best_bound = cur_bound.clone()
            best_post = post.clone()
    model.set_post(best_post)
    print(f"\nCurrent partition bound: {round(best_bound.item(), 4)}.")
    return model

def manual_coordinate_descent(model, m, bound, trainloader, loss, distribution_name):
    """
    Manual coordinate descent.
    """
    print("Manual coordinate descent...")
    best_post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
    while True:
        previous_best_bound = best_bound.clone()
        rang, factor = [], torch.max(model.get_unchanged_post()) / 100
        for i in range(len(best_post)):
            if best_post[i] >= factor:
                rang.append(i)
        np.random.shuffle(rang)
        pbar = tqdm(rang[:min(100, len(rang))])
        for i in pbar:
            for change in ['min', 'max']:
                post = model.get_unchanged_post().detach()
                new_post_i = post[i] + factor * ((change == 'max') - (change == 'min'))
                if torch.sign(post[i]) == torch.sign(new_post_i):
                    # We slightly modify the current weighting
                    post[i] = new_post_i
                    model.set_post(post)
                    # And compute the resulting bound.
                    cur_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
                    if cur_bound < best_bound:
                        best_bound = cur_bound.clone()
                        best_post = post.clone()
                    else:
                        post[i] = best_post[i]
                        model.set_post(post)
        if previous_best_bound < best_bound + 1e-2:
            break
    model.set_post(best_post.requires_grad_(True))
    print(f"\nCurrent partition bound: {round(best_bound.item(), 4)}.")
    return model


def weights_rescaling(model, m, bound, trainloader, loss, distribution_name):
    """
    We try several rescaling values for computing the optimal partition bound (the higher the rescaling factor, the
    values for b and c are, but the worse is the KL penalty).
    """
    print("Weights rescaling...")
    initial_post = model.get_unchanged_post().clone()
    best_post = model.get_unchanged_post().clone()
    n = len(best_post)
    best_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
    low = 0
    high = 10 if distribution_name in ['categorical', 'dirichlet'] else 1e2
    pbar = tqdm(range(15))
    for _ in pbar:
        try:
            mean = (low + high) / 2
            post = initial_post * mean
            model.set_post(post)
            cur_bound = compute_part_bound(model, bound, m, n, trainloader, loss, distribution_name)
            if cur_bound < best_bound:
                best_bound = cur_bound.clone()
                best_post = post.clone()
        except AssertionError:
            model.set_post(best_post)
            high -= 1
    model.set_post(best_post)
    print(f"\nCurrent partition bound: {round(best_bound.item(), 4)}.")
    return model