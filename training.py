import hydra
from pathlib import Path

from torch.utils.data import DataLoader

import wandb

from models.pretrainedDNN import LinearMultiClassifier

wandb.login()

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.deterministic_bounding import crop_weak_learners, compute_det_bound, manual_model_finetune
from core.wandb_formatting import create_config_dico, create_run_name
from core.bounds import BOUNDS
from core.losses import moment_loss, bin_loss, triple_loss, true_loss
from core.monitors import MonitorMV
from core.utils import deterministic, updating_first_seed_results, updating_last_seed_results, whether_to_run_run, \
    get_n_classes
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps
from core.Cbound.launcher import C_bound_optimization

from core.optimization import stochastic_routine, evaluate_multiset, evaluate


@hydra.main(config_path='base_config.yaml')
def main(cfg):
    whether_to_run_run(cfg)

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.training.distribution}/stmp-nt={cfg.model.stump_init}/r-n={cfg.training.rand_n}/order={cfg.bound.order}/prior={cfg.model.prior}/"

    ROOT_DIR = Path(ROOT_DIR)

    ROOT_DIR /= f"lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    if cfg.training.risk == "MC":
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"

    print("results will be saved in:", ROOT_DIR.resolve())
    distribution_name = cfg.training.distribution

    # define params for each method
    n_classes = get_n_classes(cfg.dataset)
    multiclass = n_classes > 2
    risks = { # type: (loss, bound_coeff, distribution_type, kl_factor, div)
        "Tr": (lambda x, y, z: true_loss(x, y, z, distribution_name, cfg.model.output), 1., distribution_name, 1., 'KL'),
        "FO": (lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, distribution_name, n_classes, 1, cfg.model.output), 1., distribution_name, 1., 'KL'),  # The "2" factor is taken care of later
        "SO": (lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, distribution_name, n_classes, 2, cfg.model.output), 4., distribution_name, 2., 'KL'),
        "Bin": (lambda x, y, z: bin_loss(x, y, z, cfg.model.pred, distribution_name, n_classes, cfg.training.rand_n, cfg.model.output), 2., distribution_name, cfg.training.rand_n, 'KL'),
        "Dis_Renyi": (lambda x, y, z: moment_loss(x, y, z, cfg.model.pred, distribution_name, n_classes, 1, cfg.model.output), 1., distribution_name, 1., 'Renyi'),
        "Cbound": (None, None, distribution_name, None, None),
    }

    train_errors, test_errors, train_losses, bounds, entropies, kls, times = [], [], [], [], [], [], []
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

            data = Dataset(cfg.dataset, normalize=cfg.training.normalize_data, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)
            n = len(data.X_train)

            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0), cfg.model.stump_init)

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.max_tree_depth == "None":
                    cfg.model.max_tree_depth = None

                predictors, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, samples_prop=cfg.model.samples_prop, max_depth=cfg.model.max_tree_depth, binary=data.binary, output_type=cfg.model.output)

            else:
                M = 1

            loss, bound_coeff, distribution_type, kl_factor, div = risks[cfg.training.risk]
            delta = cfg.bound.delta
            if cfg.training.risk in ["FO", "Tr"]:
                delta /= 3
            elif cfg.training.risk in ['Dis_Renyi', 'Dis_KL']:
                delta /= cfg.bound.n_grid

            bound = lambda _, model, risk, sample: BOUNDS[cfg.bound.type](n, model, risk, delta, div, False, bound_coeff, cfg.bound.order)
            test_bound = lambda _, model, risk, sample: BOUNDS['triple'](n, model, risk, delta, div, False, bound_coeff, cfg.bound.order)

            prior_coefficient = 1 / M if cfg.model.prior == "adjusted" else int(cfg.model.prior)
            prior_value = -5 if cfg.training.distribution == "categorical" else prior_coefficient

            if cfg.model.pred == "rf":
                betas = [torch.ones(M // 2) * prior_coefficient, torch.ones(M // 2) * prior_coefficient] # prior

                # weights proportional to data sizes
                model = MultipleMajorityVote(predictors, betas, n_classes, (0.5, 0.5), distribution_type,
                                             kl_factor, cfg.model.output)

            elif cfg.model.pred == "stumps-uniform":
                betas = torch.ones(M) * prior_coefficient # prior

                model = MajorityVote(predictors, betas, n_classes, distribution_type, kl_factor, cfg.model.output)
            else:
                # If needed, we reshape the images
                data.X_train = torch.tensor(data.X_train)
                data.X_test = torch.tensor(data.X_test)

                # Adding the bias
                data.X_train = torch.hstack((data.X_train, torch.ones((data.X_train.shape[0], 1))))
                data.X_test = torch.hstack((data.X_test, torch.ones((data.X_test.shape[0], 1))))

               # Finally: remove the current gradient from the examples
                data.X_train = data.X_train.detach()
                data.X_test = data.X_test.detach()

                # Here, the model corresponds in a linear layer
                input_size = data.X_train.shape[1]
                output_size = n_classes
                betas = torch.zeros(input_size, output_size)
                model = LinearMultiClassifier(input_size, output_size, betas, cfg.model.posterior_std, cfg.model.output)


            if cfg.model.pred == "rf":  # a loader per posterior
                m_train = len(data.X_train) // 2
                data.X_train = model.voters_forward([data.X_train[m_train:], data.X_train[:m_train]])

                m_test = len(data.X_test) // 2
                data.X_test = model.voters_forward([data.X_test[m_test:], data.X_test[:m_test]])

                train1 = TorchDataset(data.X_train[0], data.y_train[m_train:])
                train2 = TorchDataset(data.X_train[1], data.y_train[:m_train])

                test1 = TorchDataset(data.X_test[0], data.y_test[m_test:])
                test2 = TorchDataset(data.X_test[1], data.y_test[:m_test])

                trainloader = [
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers,
                               shuffle=True),
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers,
                               shuffle=True)
                ]

                testloader = [DataLoader(test1, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
                              DataLoader(test2, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)]
            else:
                if cfg.model.pred == "stumps-uniform":
                    data.X_train = model.voters_forward(torch.tensor(data.X_train))
                    data.X_test = model.voters_forward(torch.tensor(data.X_test))
                trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.batch_size,
                                         num_workers=cfg.num_workers, shuffle=True)
                testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096,
                                        num_workers=cfg.num_workers, shuffle=False)


            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            if cfg.training.risk == "Cbound":
                Cbound, train_error, test_error, time = C_bound_optimization(cfg, data.X_train.numpy(), data.y_train, data.X_test.numpy(), data.y_test)
                seed_results["deterministic_bound"] = Cbound
                seed_results["train-error"] = train_error
                seed_results["test-error"] = test_error
                seed_results["time"] = time
                final_bound = {'bound': Cbound}
            else:
                # First training phase
                model, final_bound, train_error, test_error, time, b_surrogate, c_surrogate = stochastic_routine(trainloader, testloader, model, optimizer, bound, cfg.bound.type, cfg.training.risk, n, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, test_bound=test_bound, distribution_name=distribution_name, n_classes=n_classes, pred_type=cfg.model.pred, compute_disintegration=cfg.training.compute_disintegration, output_type=cfg.model.output)
                if cfg.training.risk == "FO":
                    ben_bound_no_finetune, triple_bound_no_finetune, ben_triple_bound_no_finetune = compute_det_bound(model, bound, n, M, trainloader, loss, distribution_name, final_bound['bound'], b_surrogate, c_surrogate, multiclass)
                    deterministic_bound = final_bound['bound'] * 2 if cfg.training.distribution == "categorical" else 2

                    # Results are compiled in the 'seed_results' dictionary
                    seed_results = updating_first_seed_results(seed_results, time, train_error, test_error, deterministic_bound, final_bound, ben_bound_no_finetune.item(), triple_bound_no_finetune.item(), ben_triple_bound_no_finetune.item())

                    # Cropping the weight of base predictors that barely have an effect on the prediction
                    if cfg.training.distribution == "gaussian" and multiclass:
                        ben_bound_with_finetune = ben_bound_no_finetune
                        triple_bound_with_finetune = triple_bound_no_finetune
                        ben_triple_bound_with_finetune = ben_triple_bound_no_finetune
                    else:
                        model = crop_weak_learners(model, n, bound, trainloader, loss, prior_value, distribution_name)
                        model = manual_model_finetune(model, n, bound, trainloader, loss, distribution_name)
                        ben_bound_with_finetune, triple_bound_with_finetune, ben_triple_bound_with_finetune = compute_det_bound(model, bound, n, M, data, loss, distribution_name, final_bound['bound'], b_surrogate, c_surrogate)

                        # We need to recompute the train and test error
                        if multiclass:
                            val_routine = evaluate_multiset
                            test_routine = evaluate_multiset
                        else:
                            val_routine = evaluate
                            test_routine = evaluate
                        train_error = val_routine(trainloader, model)
                        test_error = test_routine(testloader, model)

                    # Results are compiled in the 'seed_results' dictionary
                    seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error, ben_bound_with_finetune.item(), triple_bound_with_finetune.item(), ben_triple_bound_with_finetune.item(), i)
                elif cfg.training.risk == "Tr":
                    optimizer = Adam(model.parameters(), lr=cfg.training.lr / 10)
                    # init learning rate scheduler
                    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
                    loss, bound_coeff, distribution_type, kl_factor, div = (lambda x, y, z: triple_loss(x, y, z, cfg.model.pred, distribution_name, n_classes, cfg.model.output), 1., distribution_name, 1., 'KL')
                    bound = lambda _, model, risk, sample: BOUNDS['triple'](n, model, risk, delta, div, False, bound_coeff, cfg.bound.order, True)
                    model, final_bound, train_error, test_error, time, b_surrogate, c_surrogate = stochastic_routine(trainloader, testloader, model, optimizer, bound, 'triple', cfg.training.risk, n, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, test_bound=test_bound, distribution_name=distribution_name, n_classes=n_classes, pred_type=cfg.model.pred, compute_disintegration=cfg.training.compute_disintegration, output_type=cfg.model.output)
                    seed_results = updating_first_seed_results(seed_results, time, train_error, test_error, final_bound['bound'], final_bound, 2, 2, 2)
                    seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error, 2, 2, 2, i)
                else:
                    seed_results = updating_first_seed_results(seed_results, time, train_error, test_error, final_bound['bound'], final_bound, 2, 2, 2)
                    seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error, 2, 2, 2, i)

            print(f"Test error: {round(seed_results['test-error'], 4)};\t Deterministic: {round(final_bound['bound'], 4)}.")

            # save seed results
            np.save(SAVE_DIR / "err-b.npy", seed_results)
            monitor.close()

            if cfg.is_using_wandb:
                wandb.log(seed_results)
                wandb.finish()

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        bounds.append(seed_results["deterministic_bound"])
        times.append(seed_results["time"])

    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times))}

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()