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

def get_b_c(possible_values, n, n_per_feature, distribution, multiclass):
    """
    Returns the lower and upper bounds of the b and c quantities.
    """
    # The triple bound is not valid for the multiclass gaussian approach
    ### We first compute the relevant values for the upper bound on b, and the lower bound on c (partition problem). ###
    if possible_values.is_cuda:
        possible_values = possible_values.cpu()
    possible_values_np = possible_values.detach().numpy()
    sum_absolute = torch.sum(torch.abs(possible_values))
    if distribution != 'gaussian':
        # Complementary voters: in the partition problem, they must be in two separate bins. Instead of treating both
        #   numbers, it's simpler to have only the absolute difference must be place in one bin.
        mins = np.min([possible_values_np[:n // 2], possible_values_np[n // 2:]], axis=0)
        changed_values = possible_values_np - np.tile(mins, 2)
        changed_values = changed_values[changed_values > 0]
    else:
        changed_values = possible_values_np
    try:
        # This function computes the partitioning problem (2 bins)
        sums = prtpy.partition(algorithm=prtpy.partitioning.ilp,
                               items=np.sort(np.abs(changed_values)),
                               objective=prtpy.obj.MaximizeSmallestSum,
                               numbins=2,
                               time_limit=5)  # time limit, in seconds (should be resolved in ~0.01 sec.)
    except ValueError:
        # In cases where the function do not converge to a solution fast enough, we avoid a crash by doing as follows
        sums = [[sum_absolute / 2], [sum_absolute / 2]]
    sum_1 = torch.sum(torch.tensor(sums[0]))
    sum_2 = torch.sum(torch.tensor(sums[1]))
    if distribution != 'gaussian':
        mins = torch.from_numpy(mins)
        sum_1 += torch.sum(mins)
        sum_2 += torch.sum(mins)

    biggest_sum = torch.max(sum_1, sum_2)
    smallest_sum = torch.min(sum_1, sum_2)

    ### We then compute the relevant values for the lower bound on b, and the upper bound on c (min/max error). ###
    if multiclass:
        min_err, max_err = 0, biggest_sum + smallest_sum
    else:
        cum_possible_values = []
        if distribution != 'gaussian':
            for j in range(n // (2 * n_per_feature)):
                cum_possible_values.append(
                    np.hstack((0, np.cumsum(
                        possible_values_np[:n // 2][int(j * n_per_feature):int((j + 1) * n_per_feature)]))) - \
                    np.flip(np.hstack((0, np.cumsum(
                        np.flip(possible_values_np[:n // 2][int(j * n_per_feature):int((j + 1) * n_per_feature)]))))) - \
                    np.flip(np.hstack((0, np.cumsum(
                        np.flip(possible_values_np[n // 2:][int(j * n_per_feature):int((j + 1) * n_per_feature)]))))) + \
                    np.hstack((0, np.cumsum(
                        possible_values_np[n // 2:][int(j * n_per_feature):int((j + 1) * n_per_feature)]))))
        else:
            for j in range(n // n_per_feature):
                cum_possible_values.append(
                    np.hstack((0, np.cumsum(possible_values_np[int(j * n_per_feature):int((j + 1) * n_per_feature)]))) - \
                    np.flip(np.hstack((0, np.cumsum(
                        np.flip(possible_values_np[int(j * n_per_feature):int((j + 1) * n_per_feature)]))))))
        min_err = 0
        max_err = 0
        for j in range(len(cum_possible_values)):
            current_min = np.inf
            current_max = -np.inf
            for k in range(n_per_feature + 1):
                if cum_possible_values[j][k] < current_min:
                    current_min = cum_possible_values[j][k]
                if cum_possible_values[j][k] > current_max:
                    current_max = cum_possible_values[j][k]
            min_err += current_min
            max_err += current_max
        if distribution != 'gaussian':
            max_err = (sum_absolute + torch.tensor(max(np.abs(max_err), np.abs(min_err)))) / 2
            min_err = sum_absolute - max_err
        else:
            max_err = torch.tensor(max(np.abs(max_err), np.abs(min_err)))
            min_err = -max_err

    ## We are now ready to compute the lower and upper bounds for both b anc c.
    if distribution == "categorical":
        return min_err, biggest_sum
    elif distribution == "dirichlet":
        return I(max_err, min_err), I(smallest_sum, biggest_sum)
    elif distribution == "gaussian":
        return Phi(max_err / n ** 0.5), Phi((smallest_sum - biggest_sum) / n ** 0.5)

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
    b, c = get_b_c(post, n, model.n_per_feature, distribution_name, multiclass)

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