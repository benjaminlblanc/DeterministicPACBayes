from copy import deepcopy
from time import time
from tqdm import tqdm
import torch

from core.deterministic_bounding import compute_bound, compute_part_triple_bound
from core.losses import triple_loss


def train_stochastic(dataloader, model, optimizer, epoch, bound=None, loss=None, monitor=None):
    """
    Training loop for when pred in [StumpsUniform, LinearClassifier].
    """
    model.train()

    last_iter = epoch * len(dataloader)

    for i, batch in enumerate(dataloader):
        n = len(batch[0])
        data = batch[1], batch[0]

        optimizer.zero_grad()

        # If there is a bound to optimize, we optimize it; otherwise, we minimize the risk
        if bound is not None:
            cost = bound(n, model, model.risk(data, loss), sample=False)
        else:
            cost = model.risk(data, loss)

        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write(last_iter + i, train={"Train-obj": cost.item()})

def train_stochastic_multiset(dataloaders, model, optimizer, epoch, bound=None, loss=None, monitor=None):
    """
    Training loop for when pred == RandomForests. This requires particularities, because each batch is separated into a
        set of predictions by a first forest, and a set of predictions by the other forest.
    """
    model.train()

    last_iter = epoch * len(dataloaders[0])

    pbar = range(len(dataloaders[0]))

    for i, *batches in zip(pbar, *dataloaders):
        X = [batch[0] for batch in batches]
        # sum sizes of loaders
        n = sum(map(len, X))
        data = [(batches[i][1], X[i]) for i in range(len(batches))]
        optimizer.zero_grad()

        if bound is not None:
            # If there is a bound to optimize, we optimize it; otherwise, we minimize the risk
            cost = bound(n, model, model.risk(data, loss), False)
        else:
            cost = model.risk(data, loss)

        cost.backward()
        optimizer.step()

        if monitor:
            monitor.write(last_iter + i, train={"Train-obj": cost.item()})


def evaluate(dataloader, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val"):
    """
    Evaluation loop for when pred in [StumpsUniform, LinearClassifier].
    """
    model.eval()

    # Depending on the situation, we either compute the regular loss or the triple loss.
    risk = torch.tensor([0., 0., 0.])
    n = torch.tensor([0, 0, 0])

    for batch in dataloader:
        data = batch[1], batch[0]

        model_risk = model.risk(data, loss=loss, mean=False)
        # If the model_risk if a tuple (of three risks and three ns), then the triple loss is computed.
        if type(model_risk) != tuple:
            risk += model_risk
            n += len(data[0])
        else:
            risk += model_risk[0]
            n += model_risk[1]

    risk /= n
    total_metrics = {"error": risk[0].item()}
    if type(model_risk) != tuple:
        risk = risk[0]

    if bounds is not None:
        for k in bounds.keys():
            if bounds[k] is not None:
                total_metrics[k] = bounds[k](n[0], model, risk, False).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def evaluate_multiset(dataloaders, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val"):
    """
    Evaluation loop for when pred in [StumpsUniform, LinearClassifier].
    """
    model.eval()

    risk = 0.
    n = 0

    for batches in zip(*dataloaders):
        X = [batch[0] for batch in batches]
        data = [(batches[i][1], X[i]) for i in range(len(batches))]
        risk += model.risk(data, loss=loss, mean=False)
        n += len(X[0])

    risk /= n
    total_metrics = {"error": risk.item()}

    if bounds is not None:
        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk, False).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def stochastic_routine(trainloader, testloader, model, optimizer, bound, n, loss,
                       monitor, lr_scheduler, test_bound, n_classes, cfg):
    """
    Main training pipeline.
    """
    bound_type = cfg.bound.type
    num_epochs = cfg.training.num_epochs
    compute_disintegration = cfg.training.compute_disintegration
    distribution_name = cfg.training.distribution
    risk_type = cfg.training.risk
    pred_type = cfg.model.pred
    output_type = cfg.model.output

    best_obj = float("inf")
    best_model = deepcopy(model)
    no_improv = 0
    best_train_stats = {bound_type: None}

    if pred_type == 'RandomForests':
        train_routine, val_routine, test_routine = train_stochastic_multiset, evaluate_multiset, evaluate_multiset
    else:
        train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate

    metric_to_optimize = bound_type if bound is not None else "error"

    # To keep track of the total training time
    t1 = time()

    pbar = tqdm(range(num_epochs))
    for e in pbar:
        no_improv += 1
        # A training epoch is completed
        train_routine(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor)
        # Just for monitoring purposes
        train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss, monitor=monitor, tag="train")

        pbar.set_description("train obj %s" % train_stats[metric_to_optimize])
        # If there are improvements, we update the best model
        if train_stats[metric_to_optimize] < best_obj:
            best_obj = train_stats[metric_to_optimize]
            best_train_stats = train_stats
            best_model = deepcopy(model)
            no_improv = 0
        if lr_scheduler:
            lr_scheduler.step(train_stats[metric_to_optimize])
        # This is the criteria for early stopping
        if no_improv == num_epochs // 4:
            break
    t2 = time()

    val_routine(trainloader, model, loss=loss)
    test_error = test_routine(testloader, best_model)
    string = f"Test error: {round(test_error['error'], 4)}"
    if metric_to_optimize == "error":
        train_error = best_train_stats["error"]
        final_bound = {'bound': 1}
        string += "\n"
    else:
        train_error = val_routine(trainloader, best_model)
        final_bound = {'bound': round(best_obj, 4)}
        string += f"; {bound_type} bound: {round(best_train_stats[bound_type], 4)}"

    if risk_type == "FO":
        triple_bnd = compute_bound(model, test_bound, n, trainloader, lambda x, y, z: triple_loss(x, y, z, pred_type, distribution_name, n_classes, output_type), False)
        M = torch.prod(torch.tensor(model.get_unchanged_post().shape))
        partition_bound = compute_part_triple_bound(model, bound, n, M, trainloader, loss, distribution_name)[2]
        string += f"; partition bound: {round(partition_bound.item(), 4)}\n"
    else:
        triple_bnd = (None, None)
        string += "\n"
    print(string)

    # The disintegration can only be computed for two different types of risk: FO (yielding KL_disintegration) and
    #   Dis_Renyi (yielding Renyi_disintegration).
    if risk_type in ['FO', 'Dis_Renyi'] and compute_disintegration:
        print("Computing the average stats. over 20 random posterior draw...")
        test_errors = []
        bounds = []
        model_to_randomize = deepcopy(best_model)
        best_model_post = best_model.get_post()
        for i in range(20):
            # We first draw a new posterior and assign it
            model_to_randomize.random_draw_new_post()
            # We then computed the several considered metrics
            test_errors.append(test_routine(testloader, model_to_randomize)['error'])
            bounds.append(compute_bound(model_to_randomize, bound, n, trainloader, loss, disintegrated=True))
            # And we set back the optimal posterior, so that the draw is done according to the correct distribution
            model_to_randomize.set_post(best_model_post)
        test_error['error_sampled'] = torch.mean(torch.tensor(test_errors)).item()
        test_error['error_sampled_std'] = torch.std(torch.tensor(test_errors)).item()
        final_bound['bound_sampled'] = torch.mean(torch.tensor(bounds)).item()
        final_bound['bound_sampled_std'] = torch.std(torch.tensor(bounds)).item()
    else:
        test_error['error_sampled'] = 0
        test_error['error_sampled_std'] = 0
        final_bound['bound_sampled'] = 0
        final_bound['bound_sampled_std'] = 0

    return best_model, final_bound, train_error, test_error, t2 - t1, triple_bnd[0], triple_bnd[1]