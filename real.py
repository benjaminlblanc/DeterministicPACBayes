import hydra
from pathlib import Path
import wandb
wandb.login()

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.deterministic_bounding import crop_weak_learners, compute_det_bound, manual_model_finetune
from core.wandb_formatting import create_config_dico, create_run_name
from core.bounds import BOUNDS
from core.losses import moment_loss, bin_loss
from core.monitors import MonitorMV
from core.utils import deterministic, updating_first_seed_results, updating_last_seed_results, whether_to_run_run, \
    get_n_classes
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

from optimization import stochastic_routine


@hydra.main(config_path='config/real.yaml')
def main(cfg):
    whether_to_run_run(cfg)

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/{cfg.training.distribution}/optimize-bound={cfg.training.opt_bound}/{cfg.model.pred}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior={cfg.model.prior}/"

    ROOT_DIR = Path(ROOT_DIR)

    if cfg.model.uniform:
        ROOT_DIR /= "uniform"
    else:
        ROOT_DIR /= f"lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    if cfg.training.risk == "MC":
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"

    print("results will be saved in:", ROOT_DIR.resolve())
    distribution_name = cfg.training.distribution

    # define params for each method
    n_classes = get_n_classes(cfg.dataset)
    risks = { # type: (loss, bound_coeff, distribution_type, kl_factor, div)
        "FO": (lambda x, y, z: moment_loss(x, y, z, distribution_name, n_classes, order=1), 1., distribution_name, 1., 'KL'),  # The "2" factor is taken care of later
        "SO": (lambda x, y, z: moment_loss(x, y, z, distribution_name, n_classes, order=2), 4., distribution_name, 2., 'KL'),
        "Bin": (lambda x, y, z: bin_loss(x, y, z, distribution_name, n_classes, n=cfg.training.rand_n), 2., distribution_name, cfg.training.rand_n, 'KL'),
        "Dis_Renyi": (lambda x, y, z: moment_loss(x, y, z, distribution_name, n_classes, order=1), 1., distribution_name, 1., 'Renyi'),
    }

    train_errors, test_errors, train_losses, bounds, strengths, entropies, kls, times = [], [], [], [], [], [], [], []
    for i in range(cfg.num_trials):

        current_seed = cfg.training.seed + i
        print("seed", current_seed)
        deterministic(current_seed)

        SAVE_DIR = ROOT_DIR / f"seed={current_seed}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (SAVE_DIR / "err-b.npy").is_file():
            print(SAVE_DIR)
            # load saved stats
            seed_results = np.load(SAVE_DIR / "err-b.npy", allow_pickle=True).item()

        else:
            config = create_config_dico(cfg)
            run_name = create_run_name(config, current_seed)
            if cfg.is_using_wandb:
                wandb.init(name=str(run_name), project=cfg.project_name, config=config)

            seed_results = {}

            try:
                data = Dataset(cfg.dataset.distr, n_train=cfg.dataset.N_train, n_test=cfg.dataset.N_test, noise=cfg.dataset.noise)
            except:
                data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)

            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0), cfg.model.stump_init)

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                predictors, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth, binary=data.binary)

            else:
                raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")

            loss, coeff, distr, kl_factor, div = risks[cfg.training.risk]
            a = cfg.model.a
            delta = cfg.bound.delta
            if cfg.training.risk in ['Dis_Renyi', 'Dis_KL']:
                delta /= cfg.bound.n_grid

            bound = None
            if cfg.training.opt_bound:

                print(f"Optimize {cfg.bound.type} bound")

                if cfg.bound.stochastic:

                    print("Evaluate bound regularizations over mini-batch")
                    bound = lambda n, model, risk, sample: BOUNDS[cfg.bound.type](n, model, risk, delta, div, False, coeff, cfg.bound.order)

                else:
                    print("Evaluate bound regularizations over whole training set")
                    n = len(data.X_train)
                    bound = lambda _, model, risk, sample: BOUNDS[cfg.bound.type](n, model, risk, delta, div, False, coeff, cfg.bound.order)

            if cfg.model.pred == "rf": # a loader per posterior

                data.X_train = data.X_train[:10000]
                data.y_train = data.y_train[:10000]

                m_train = len(data.X_train) // 2
                train1 = TorchDataset(data.X_train[m_train:], data.y_train[m_train:])
                train2 = TorchDataset(data.X_train[:m_train], data.y_train[:m_train])
                trainloader = [
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                ]

            else:
                train = TorchDataset(data.X_train, data.y_train)
                trainloader = DataLoader(train, batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)

            testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

            prior_coefficient = 1 / M if cfg.model.prior == "adjusted" else int(cfg.model.prior)
            prior_value = -5 if cfg.training.distribution == "categorical" else prior_coefficient

            if cfg.model.pred == "rf":
                betas = [torch.ones(M) * prior_coefficient for _ in predictors] # prior

                # weights proportional to data sizes
                model = MultipleMajorityVote(predictors, betas, a, weights=(0.5, 0.5), distr=distr, kl_factor=kl_factor)

            else:
                betas = torch.ones(M) * prior_coefficient # prior

                model = MajorityVote(predictors, betas, a, distr=distr, kl_factor=kl_factor)

            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            # First training phase
            model, final_bound, _, train_error, test_error, time = stochastic_routine(trainloader, testloader, model, optimizer, bound, cfg.bound.type, cfg.training.risk, n, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, true_risk_bounding=False)
            if cfg.training.risk == "FO":
                if cfg.training.distribution == "gaussian" and n_classes > 2:
                    ben_bound_no_finetune = 1
                else:
                    ben_bound_no_finetune = compute_det_bound(model, bound, n, M, trainloader, loss, distribution_name, cur_PB_bound=final_bound['bound']).item()
                if cfg.training.distribution == 'categorical':
                    deterministic_bound = final_bound['bound'] * 2
                else:
                    deterministic_bound = 2

                # Results are compiled in the 'seed_results' dictionary
                seed_results = updating_first_seed_results(seed_results, time, model, train_error, test_error, deterministic_bound, final_bound, ben_bound_no_finetune)

                # Cropping the weight of base predictors that barely have an effect on the prediction
                if seed_results["factor_no_finetune"] < 2:
                    model = crop_weak_learners(model, n, bound, trainloader, loss, prior_value, distribution_name)
                    model = manual_model_finetune(model, n, bound, trainloader, loss, distribution_name)

                    # Second training phase
                    model, final_bound, _, train_error, test_error, time = stochastic_routine(trainloader, testloader, model, optimizer, bound, cfg.bound.type, cfg.training.risk, n, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, true_risk_bounding=True)
                if cfg.training.distribution == "gaussian" and n_classes > 2:
                    ben_bound_with_finetune = 1
                else:
                    ben_bound_with_finetune = compute_det_bound(model, bound, n, M, data, loss, distribution_name, cur_PB_bound=final_bound['bound']).item()

                # Results are compiled in the 'seed_results' dictionary
                seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune, i)
            else:
                deterministic_bound, ben_bound_no_finetune, ben_bound_with_finetune = final_bound['bound'], 2, 2
                seed_results = updating_first_seed_results(seed_results, time, model, train_error, test_error, deterministic_bound, final_bound, ben_bound_no_finetune)
                seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune, i)

            print(f"Test error: {round(seed_results['test-error'], 4)};\t Deterministic: {round(final_bound['bound'], 4)};\t Factor: {round(seed_results['factor_no_finetune'], 4)};\t Factor (finetuned): {round(seed_results['factor_with_finetune'], 4)}")

            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)
            monitor.close()

            if cfg.is_using_wandb:
                wandb.log(seed_results)
                wandb.finish()

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        kls.append(seed_results["KL"])
        bounds.append(seed_results["deterministic_bound"])
        times.append(seed_results["time"])

    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times)), "KL": (np.mean(kls), np.std(kls))}

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()