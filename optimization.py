from copy import deepcopy
from time import time
from tqdm import tqdm

from core.deterministic_bounding import compute_det_bound
from models.majority_vote import MultipleMajorityVote


def train_batch(n, data, model, optimizer, bound=None, loss=None, nb_iter=1e4, monitor=None, true_risk_bounding=False):

    model.train()
    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:
        optimizer.zero_grad()

        n_alphas = len(model.post)

        if bound is not None:
            if true_risk_bounding:
                cost = compute_det_bound(model, bound, n, n_alphas, data, loss, model.distribution_name)
            else:
                cost = bound(n, model, model.risk(data, loss))

        else:
            cost = model.risk(data, loss)

        pbar.set_description("train obj %s" % cost.item())
        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write_all(i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})


def train_stochastic(dataloader, model, optimizer, epoch, bound=None, loss=None, monitor=None, true_risk_bounding=False):

    model.train()

    last_iter = epoch * len(dataloader)
    train_obj = 0.

    for i, batch in enumerate(dataloader):

        n = len(batch[0])
        data = batch[1], model(batch[0])

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        n_alphas = len(model.post)

        if bound is not None:
            if true_risk_bounding:
                cur_PB_bound = bound(n, model, model.risk(data, loss))
                cost = compute_det_bound(model, bound, n, n_alphas, data, loss, model.distribution_name, cur_PB_bound)
            else:
                cost = bound(n, model, model.risk(data, loss))

        else:
            cost = model.risk(data, loss)

        train_obj += cost.item()

        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write_all(last_iter + i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})

def train_stochastic_multiset(dataloaders, model, optimizer, epoch, bound=None, loss=None, monitor=None, true_risk_bounding=False):
    model.train()

    last_iter = epoch * len(dataloaders[0])
    train_obj = 0.

    pbar = range(len(dataloaders[0]))

    for i, *batches in zip(pbar, *dataloaders):
        X = [batch[0] for batch in batches]
        # sum sizes of loaders
        n = sum(map(len, X))
        pred = model(X)
        data = [(batches[i][1], pred[i]) for i in range(len(batches))]
        optimizer.zero_grad()
        n_alphas = len(model.get_post())

        if bound is not None:
            if true_risk_bounding:
                cost = compute_det_bound(model, bound, n, n_alphas, data, loss, model.distribution_name)
            else:
                cost = bound(n, model, model.risk(data, loss))

        else:
            cost = model.risk(data, loss)

        train_obj += cost.item()
        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write_all(last_iter + i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})


def evaluate(dataloader, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val"):
    model.eval()

    risk = 0.
    strength = 0.
    n = 0

    for batch in dataloader:
        data = batch[1], model(batch[0])
        risk += model.risk(data, loss=loss, mean=False)
        strength += sum(model.voter_strength(data))
        n += len(data[0])

    risk /= n
    strength /= n
    total_metrics = {"error": risk.item(), "strength": strength.item()}

    if bounds is not None:
        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def evaluate_multiset(dataloaders, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val"):
    model.eval()

    risk = 0.
    n = 0
    strength = 0.

    for batches in zip(*dataloaders):
        X = [batch[0] for batch in batches]
        pred = model(X)
        data = [(batches[i][1], pred[i]) for i in range(len(batches))]

        risk += model.risk(data, loss=loss, mean=False)
        strength += sum(model.voter_strength(data))

        n += len(X[0])

    risk /= n
    strength /= n
    total_metrics = {"error": risk.item(), "strength": strength.item()}

    if bounds is not None:

        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def stochastic_routine(trainloader, testloader, model, optimizer, bound, bound_type, loss=None, monitor=None,
                       num_epochs=100, lr_scheduler=None, true_risk_bounding=False):

    best_bound = float("inf")
    best_model = deepcopy(model)
    no_improv = 0
    best_train_stats = {bound_type: None}

    if isinstance(model, MultipleMajorityVote):  # then expect multiple dataloaders
        train_routine = train_stochastic_multiset
        val_routine = evaluate_multiset
        test_routine = lambda d, *args, **kwargs: evaluate_multiset((d, d), *args, **kwargs)
    else:
        train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate

    t1 = time()

    pbar = tqdm(range(num_epochs))
    for e in pbar:
        train_routine(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor, true_risk_bounding=true_risk_bounding)

        train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss, monitor=monitor, tag="train")  # just for monitoring purposes

        no_improv += 1
        if train_stats[bound_type] < best_bound:
            best_bound = train_stats[bound_type]
            best_train_stats = train_stats
            best_model = deepcopy(model)
            no_improv = 0

        # reduce learning rate if needed
        if lr_scheduler:
            lr_scheduler.step(train_stats[bound_type])

        if no_improv == num_epochs // 4:
            break

        pbar.set_description("train obj %s" % train_stats[bound_type])

    t2 = time()

    train_error = val_routine(trainloader, best_model)
    test_error = test_routine(testloader, best_model)

    print(f"Test error: {test_error['error']}; {bound_type} bound: {best_train_stats[bound_type]}\n")

    return best_model, best_bound, best_train_stats, train_error, test_error, t2 - t1