import hydra
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
wandb.login()

from data.init import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.pretrainedDNN import LinearMultiClassifier

from core.losses import initialize_risk
from core.deterministic_bounding import clip_weak_learners, compute_part_bound, manual_coordinate_descent, \
    weights_rescaling
from core.wandb_formatting import create_config_dico, create_run_name
from core.bounds import BOUNDS
from core.monitors import MonitorMV
from core.utils import *
from core.Cbound.launcher import C_bound_optimization
from core.optimization import stochastic_routine, evaluate_multiset, evaluate


@hydra.main(config_path='base_config.yaml')
def main(cfg):
    # This function tests whether the hyperparameters are coherent with each others.
    whether_to_run_run(cfg)

    # For each run, there exists a unique repository.
    ROOT_DIR = create_root_dir(cfg)
    ROOT_DIR = Path(ROOT_DIR)
    print("Results will be saved in: ", ROOT_DIR.resolve(), ".")

    train_errors, test_errors, bounds, times = [], [], [], []
    # We iterate for as many trials as required, incrementing the random seed by 1 for each one.
    for i in range(cfg.num_trials):
        current_seed = cfg.training.seed + i
        print(f"Current seed: {current_seed}.")

        # Each package has its random seed set according to the current seed.
        deterministic(current_seed)

        SAVE_DIR = ROOT_DIR / f"seed={current_seed}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (SAVE_DIR / "err-b.npy").is_file():
            # If results are already saved in that folder, the run has already been done.
            print("Run already done, skipping...")

            # The aggregated results are later computed for the corresponding number of trials.
            seed_results = np.load(SAVE_DIR / "err-b.npy", allow_pickle=True).item()

        else:
            seed_results = {}
            if cfg.is_using_wandb:
                # If using WandB, then the hyperparameters must be consigned in a particular way.
                config = create_config_dico(cfg)
                run_name = create_run_name(config, current_seed)
                wandb.init(name=str(run_name), project=cfg.project_name, config=config)

            # Initialize the dataset, loss utilized, etc.
            data = Dataset(cfg.dataset, normalize=cfg.training.normalize_data,
                           data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)
            m_train = len(data.X_train)

            # Define params for each method.
            n_classes = get_n_classes(cfg.dataset)
            multiclass = n_classes > 2
            loss, bound_coeff, kl_factor, div = initialize_risk(cfg, n_classes)
            distribution_name = cfg.training.distribution

            delta = cfg.bound.delta
            if cfg.training.risk == 'Dis_Renyi':
                # If risk == Dis_Renyi, we account for the fact that n_grid values for "order" were considered.
                delta /= cfg.bound.n_grid

            # The main bound to optimize, and the surrogate bound to test (if risk == FO).
            validloader, trtestloader = None, None
            if cfg.training.risk in ['Test', 'VCdim']:
                bound = None
            else:
                bound = lambda _, modl, risk, sample: BOUNDS[cfg.bound.type](m_train, model, risk, delta, div, False,
                                                                              bound_coeff, cfg.bound.order)
            # Initializing the predictors, the number of base classifiers.
            predictors, n = initialize_predictors(cfg, data)
            # This corresponds to the prior value to consider for each parameter.
            prior_coefficient = 1 / n if cfg.model.prior == "adjusted" else int(cfg.model.prior)
            # This corresponds to the replacement value to use in the "crop_weak_learner" method.
            prior_value = -5 if cfg.training.distribution == "categorical" else prior_coefficient

            if cfg.model.pred == "UniformStumps":
                betas = torch.ones(n) * prior_coefficient  # uniform prior

                model = MajorityVote(predictors, betas, n_classes, cfg.model.n, distribution_name, kl_factor, cfg.model.output)

                # We change the dataset by the model prediction, since the stumps won't change anymore.
                data.X_train = model.forward(torch.tensor(data.X_train))
                data.X_test = model.forward(torch.tensor(data.X_test))

                if cfg.training.risk == "Test":
                    tr_split = int(cfg.training.splits[0] * m_train)
                    vd_split = int((cfg.training.splits[0] + cfg.training.splits[1]) * m_train)
                    X_train, y_train = data.X_train[:tr_split], data.y_train[:tr_split]
                    X_valid, y_valid = (data.X_train[tr_split:vd_split], data.y_train[tr_split:vd_split])
                    X_trtest, y_trtest = data.X_train[vd_split:], data.y_train[vd_split:]
                    trainloader = DataLoader(TorchDataset(X_train, y_train),
                                             batch_size=cfg.training.batch_size,
                                             num_workers=cfg.num_workers, shuffle=True)
                    validloader = DataLoader(TorchDataset(X_valid, y_valid),
                                             batch_size=cfg.training.batch_size,
                                             num_workers=cfg.num_workers, shuffle=True)
                    trtestloader = DataLoader(TorchDataset(X_trtest, y_trtest),
                                               batch_size=cfg.training.batch_size,
                                               num_workers=cfg.num_workers, shuffle=True)
                    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096,
                                            num_workers=cfg.num_workers, shuffle=False)
                else:
                    trainloader = DataLoader(TorchDataset(data.X_train, data.y_train),
                                             batch_size=cfg.training.batch_size,
                                             num_workers=cfg.num_workers, shuffle=True)
                    testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096,
                                            num_workers=cfg.num_workers, shuffle=False)

            elif cfg.model.pred == "RandomForests": # random forest
                betas = [torch.ones(n // 2) * prior_coefficient, torch.ones(n // 2) * prior_coefficient] # uniform prior

                # weights equivalent for each forest
                model = MultipleMajorityVote(predictors, betas, n_classes, cfg.model.n, (0.5, 0.5),
                                             distribution_name, kl_factor, cfg.model.output)

                if cfg.training.risk == "Test":
                    tr_split = int(cfg.training.splits[0] * m_train)
                    vd_split = int((cfg.training.splits[0] + cfg.training.splits[1]) * m_train)
                    X_train, y_train = data.X_train[:tr_split], data.y_train[:tr_split]
                    X_valid, y_valid = (data.X_train[tr_split:vd_split], data.y_train[tr_split:vd_split])
                    X_trtest, y_trtest = data.X_train[vd_split:], data.y_train[vd_split:]
                    half_m_train = len(X_train) // 2
                    half_m_valid = len(X_valid) // 2
                    half_m_trtest = len(X_trtest) // 2
                    half_m_test = len(data.X_test) // 2

                    # We change the dataset by the model prediction, since the trees won't change anymore.
                    X_train = model.forward([X_train[half_m_train:], X_train[:half_m_train]])
                    X_valid = model.forward([X_valid[half_m_valid:], X_valid[:half_m_valid]])
                    X_trtest = model.forward([X_trtest[half_m_trtest:], X_trtest[:half_m_trtest]])
                    X_test = model.forward([data.X_test[half_m_test:], data.X_test[:half_m_test]])

                    train1 = TorchDataset(X_train[0], y_train[half_m_train:])
                    train2 = TorchDataset(X_train[1], y_train[:half_m_train])

                    valid1 = TorchDataset(X_valid[0], y_valid[half_m_valid:])
                    valid2 = TorchDataset(X_valid[1], y_valid[:half_m_valid])

                    trtest1 = TorchDataset(X_trtest[0], y_trtest[half_m_trtest:])
                    trtest2 = TorchDataset(X_trtest[1], y_trtest[:half_m_trtest])

                    test1 = TorchDataset(X_test[0], data.y_test[half_m_test:])
                    test2 = TorchDataset(X_test[1], data.y_test[:half_m_test])

                    trainloader = [
                        DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                        DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                    ]

                    validloader = [
                        DataLoader(valid1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                        DataLoader(valid2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                    ]

                    trtestloader = [
                        DataLoader(trtest1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                        DataLoader(trtest2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                    ]

                    testloader = [DataLoader(test1, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
                                  DataLoader(test2, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)]
                else:
                    half_m_train = len(data.X_train) // 2
                    tq_m_train = int(len(data.X_train) * (3 / 4))
                    half_m_test = len(data.X_test) // 2
                    tq_m_test = int(len(data.X_test) * (3 / 4))
                    if cfg.training.risk == "VCdim":
                        # We change the dataset by the model prediction, since the trees won't change anymore.
                        data.X_train = model.forward([data.X_train[tq_m_train:], data.X_train[half_m_train:tq_m_train]])
                        data.X_test = model.forward([data.X_test[tq_m_test:], data.X_test[half_m_test:tq_m_test]])
                    else:
                        # We change the dataset by the model prediction, since the trees won't change anymore.
                        data.X_train = model.forward([data.X_train[half_m_train:], data.X_train[:half_m_train]])
                        data.X_test = model.forward([data.X_test[half_m_test:], data.X_test[:half_m_test]])

                    train1 = TorchDataset(data.X_train[0], data.y_train[half_m_train:])
                    train2 = TorchDataset(data.X_train[1], data.y_train[:half_m_train])

                    test1 = TorchDataset(data.X_test[0], data.y_test[half_m_test:])
                    test2 = TorchDataset(data.X_test[1], data.y_test[:half_m_test])

                    trainloader = [
                        DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers,
                                   shuffle=True),
                        DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers,
                                   shuffle=True)
                    ]

                    testloader = [DataLoader(test1, batch_size=4096, num_workers=cfg.num_workers, shuffle=False),
                                  DataLoader(test2, batch_size=4096, num_workers=cfg.num_workers, shuffle=False)]

            else:
                # Adding the bias
                data.X_train = torch.hstack((torch.tensor(data.X_train), torch.ones((data.X_train.shape[0], 1))))
                data.X_test = torch.hstack((torch.tensor(data.X_test), torch.ones((data.X_test.shape[0], 1))))

                input_size = len(data.X_train[0])
                output_size = n_classes
                # Here, the model corresponds in a linear layer; not a vector of voters, but a matrix of weights
                betas = torch.zeros(input_size, output_size)  # uniform prior
                n = input_size * output_size

                # Here, no need to compute a forward pass on the dataset: it directly corresponds to the embedding
                model = LinearMultiClassifier(input_size, output_size, betas, cfg.model.posterior_std, cfg.model.output)

                trainloader = DataLoader(TorchDataset(data.X_train, data.y_train), batch_size=cfg.training.batch_size,
                                         num_workers=cfg.num_workers, shuffle=True)
                testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096,
                                        num_workers=cfg.num_workers, shuffle=False)

            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            # The Cbound algorithm has a training pipeline of his own.
            if cfg.training.risk == "Cbound":
                cbound, tr_err, te_err, time = C_bound_optimization(cfg, data.X_train.numpy(), data.y_train,
                                                                         data.X_test.numpy(), data.y_test)
                seed_results["deterministic_bound"] = cbound
                seed_results["train-error"] = tr_err
                seed_results["train-error_finetune"] = tr_err
                seed_results["test-error"] = te_err
                seed_results["test-error_finetune"] = te_err
                seed_results["time"] = time
            else:
                # First training phase
                model, final_bound, train_error, test_error, time = \
                    stochastic_routine(trainloader, validloader, trtestloader, testloader, model, optimizer, bound,
                                       m_train, loss, monitor, lr_scheduler, n_classes, cfg)
                # part_bnd refers to the partition bound.
                #   deterministic_bound refers to the benchmark bound (factor 2, in the case of risk = FO).
                #   The proposed bounds are optimized with risk = FO, since the objective function is similar to that
                #   of the factor-2 bound.
                if cfg.training.risk == "FO":
                    part_bnd = compute_part_bound(model, bound, m_train, n, trainloader, loss, distribution_name,
                                                  final_bound['bound'], multiclass)
                    # The dirichlet distribution is not allowed for the benchmarks algorithms
                    deterministic_bound = final_bound['bound'] * 2 if cfg.training.distribution != "dirichlet" else 2

                    # Results are compiled in the 'seed_results' dictionary
                    seed_results = updating_first_seed_results(seed_results, time, train_error, test_error,
                                                               deterministic_bound, final_bound, part_bnd.item())

                    if cfg.training.distribution == "gaussian" and multiclass:
                        # The partition bound does not concern the multiclass gaussian approach
                        part_bnd_tnd = part_bnd
                    else:
                        # Cropping the weight of base predictors that barely have an effect on the prediction
                        model = clip_weak_learners(model, m_train, bound, trainloader, loss, prior_value,
                                                   distribution_name)

                        # Manual coordinate descent on the weights
                        model = manual_coordinate_descent(model, m_train, bound, trainloader, loss, distribution_name)

                        # Finally: we try finding the optimal weight scaling for the partition bound.
                        model = weights_rescaling(model, m_train, bound, trainloader, loss, distribution_name)

                        # We need to recompute the train error, test error, and the bounds
                        part_bnd_tnd = compute_part_bound(model, bound, m_train, n, trainloader, loss,
                                                          distribution_name, None, multiclass)
                        if multiclass:
                            val_routine = evaluate_multiset
                            test_routine = evaluate_multiset
                        else:
                            val_routine = evaluate
                            test_routine = evaluate
                        train_error = val_routine(trainloader, model)
                        test_error = test_routine(testloader, model)

                    # Results are compiled in the 'seed_results' dictionary
                    seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error,
                                                              part_bnd_tnd.item(), i)
                else:
                    seed_results = updating_first_seed_results(seed_results, time, train_error, test_error,
                                                               final_bound['bound'], final_bound, part_bnd=2)
                    seed_results = updating_last_seed_results(seed_results, cfg, train_error, test_error,
                                                              part_bnd_tnd=2, i=i)

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

    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),
               "test-error": (np.mean(test_errors), np.std(test_errors)),
               cfg.bound.type: (np.mean(bounds), np.std(bounds)),
               "time": (np.mean(times), np.std(times))}

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()