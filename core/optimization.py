from copy import deepcopy
from time import time
from tqdm import tqdm
import torch

from core.deterministic_bounding import compute_det_bound, compute_bound
from core.losses import triple_loss
from models.majority_vote import MultipleMajorityVote


def train_stochastic(dataloader, model, optimizer, epoch, bound=None, loss=None, monitor=None):

    model.train()

    last_iter = epoch * len(dataloader)

    for i, batch in enumerate(dataloader):

        n = len(batch[0])
        data = batch[1], model(batch[0])

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss), sample=False)

        else:
            cost = model.risk(data, loss)

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
                cost = compute_det_bound(model, bound, n, n_alphas, data, loss, model.distribution_name)[2]
            else:
                cost = bound(n, model, model.risk(data, loss), False)

        else:
            cost = model.risk(data, loss)

        train_obj += cost.item()
        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write_all(last_iter + i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})


def evaluate(dataloader, model, epoch=-1, bounds=None, loss=None, monitor=None, centered=True, tag="val"):
    model.eval()

    risk = 0.
    risk_multi = torch.tensor([0., 0., 0.])
    n, n_1, n_2, n_3 = 0, 0, 0, 0

    for batch in dataloader:
        data = batch[1], model(batch[0])
        model_risk = model.risk(data, loss=loss, mean=False, centered=centered)
        if type(model_risk) != tuple:
            risk += model_risk
            n += len(data[0])
        else:
            risk_multi[0] += model_risk[0][0]
            risk_multi[1] += model_risk[0][1]
            risk_multi[2] += model_risk[0][2]
            n_1 += model_risk[1][0]
            n_2 += model_risk[1][1]
            n_3 += model_risk[1][2]

    if type(model_risk) != tuple:
        risk /= n
        total_metrics = {"error": risk.item()}
    else:
        risk_multi[0] /= n_1
        risk_multi[1] /= n_2
        risk_multi[2] /= n_3
        risk = risk_multi
        total_metrics = {"error": risk[0].item()}

    if bounds is not None:
        for k in bounds.keys():
            if bounds[k] is not None:
                total_metrics[k] = bounds[k](n, model, risk, False).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def evaluate_multiset(dataloaders, model, epoch=-1, bounds=None, loss=None, monitor=None, centered=True, tag="val"):
    model.eval()

    risk = 0.
    n = 0

    for batches in zip(*dataloaders):
        X = [batch[0] for batch in batches]
        try:
            pred = model(X)
            data = [(batches[i][1], pred[i]) for i in range(len(batches))]
            risk += model.risk(data, loss=loss, mean=False, centered=centered)
        except RuntimeError:
            pred = model.voters_forward(X)
            data = [(batches[i][1], pred[i]) for i in range(len(batches))]
            risk += model.risk(data, loss=loss, mean=False, centered=centered)
        n += len(X[0])

    risk /= n
    total_metrics = {"error": risk.item()}

    if bounds is not None:

        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk, False).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def stochastic_routine(trainloader, testloader, model, optimizer, bound, bound_type, risk_type, n, loss=None, monitor=None,
                       num_epochs=100, lr_scheduler=None, test_bound=None, distribution_name='', n_classes=0, pred_type='rf', compute_dis=False):

    best_bound = float("inf")
    best_model = deepcopy(model)
    no_improv = 0
    best_train_stats = {bound_type: None}

    if isinstance(model, MultipleMajorityVote):  # then expect multiple dataloaders
        train_routine = train_stochastic_multiset
        val_routine = evaluate_multiset
        test_routine = evaluate_multiset
    else:
        train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate

    t1 = time()

    pbar = tqdm(range(num_epochs))
    for e in pbar:
        train_routine(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)

        train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss, monitor=monitor, tag="train")  # just for monitoring purposes
        no_improv += 1
        if bound is not None:
            if train_stats[bound_type] < best_bound:
                best_bound = train_stats[bound_type]
                best_train_stats = train_stats
                best_model = deepcopy(model)
                no_improv = 0

            # reduce learning rate if needed
            if lr_scheduler:
                lr_scheduler.step(train_stats[bound_type])

            pbar.set_description("train obj %s" % train_stats[bound_type])
        else:
            if train_stats["error"] < best_bound:
                best_bound = train_stats["error"]
                best_train_stats = train_stats
                best_model = deepcopy(model)
                no_improv = 0
            if lr_scheduler:
                lr_scheduler.step(train_stats["error"])

            pbar.set_description("train obj %s" % train_stats["error"])

        if no_improv == num_epochs // 4:
            break
    print()
    t2 = time()

    train_error = val_routine(trainloader, best_model)
    test_error = test_routine(testloader, best_model)
    final_bound = {'bound': best_bound}

    if risk_type == "FO":
        triple_bnd = compute_bound(model, test_bound, n, trainloader, lambda x, y, z: triple_loss(x, y, z, pred_type, distribution_name, n_classes), False)
    else:
        triple_bnd = (None, None, None)

    if risk_type in ['FO', 'Dis_Renyi'] and compute_dis:
        test_errors = []
        bounds = []
        model_to_try = deepcopy(best_model)
        best_model_post = best_model.get_post()
        for i in range(20):
            model_to_try.random_draw_new_post()
            test_errors.append(test_routine(testloader, model_to_try, centered=False)['error'])
            bounds.append(compute_bound(model_to_try, bound, n, trainloader, loss, True))
            model_to_try.set_post(best_model_post)
        test_error['error_sampled'] = torch.mean(torch.tensor(test_errors)).item()
        test_error['error_sampled_std'] = torch.std(torch.tensor(test_errors)).item()
        final_bound['bound_sampled'] = torch.mean(torch.tensor(bounds)).item()
        final_bound['bound_sampled_std'] = torch.std(torch.tensor(bounds)).item()
    else:
        test_error['error_sampled'] = 0
        test_error['error_sampled_std'] = 0
        final_bound['bound_sampled'] = 0
        final_bound['bound_sampled_std'] = 0

    string = f"Test error: {test_error['error']}"
    if bound is not None:
        string += f"; {bound_type} bound: {best_train_stats[bound_type]}\n"
    else:
        string += "\n"
    print(string)

    return best_model, final_bound, train_error, test_error, t2 - t1, triple_bnd[0], triple_bnd[1]