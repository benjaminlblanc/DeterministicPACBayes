import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def csv_to_latex(file_name, hyperparameters, type, show):
    """
    Generates a latex result table, given a csv file containing the result, imported from WandB.
    """
    if type == 'regular':
        numerical_results = ['test-error_finetune', 'test-error']  # Do not change that order
        metrics = ['part_bnd_tnd', 'deterministic_bound']
        last_distinctions = ['pred', 'dataset', 'risk']  # Do not change that order
    elif type == 'error_min':
        numerical_results = ['test-error_finetune', 'part_bnd_tnd']  # Do not change that order
        metrics = ['train-error', 'deterministic_bound']
        last_distinctions = ['pred', 'dataset', 'risk']  # Do not change that order
    elif type == 'distributions':
        numerical_results = ['test-error_finetune', 'test-error']  # Do not change that order
        metrics = ['part_bnd_tnd']
        last_distinctions = ['pred', 'dataset', 'distribution']  # Do not change that order
    else:
        assert False

    # We first import the data
    try:
        with open(file_name, 'r') as det_ :
            det = [line.replace('"', '').replace('\n', '').split(',') for line in det_]
    except FileNotFoundError:
        assert False, ("Make sure to download a WandB results csv file, to put it in the 'graphics' folder, and to call"
                       " the correct name.")
    det_.close()
    initial_df = pd.DataFrame(det[1:], columns=det[0])
    initial_df.replace('', 1, inplace=True)

    # We transform the numerical entities to floats from strings
    for numerical in numerical_results + metrics:
        initial_df = initial_df.astype({numerical: float})

    # We create the dictionnary containing the relevant aggregated results
    final_results = {}

    for i in range(len(metrics)):
        # We only keep the relevant column values
        df = initial_df[hyperparameters + [metrics[i]] + numerical_results]

        # We create the average and std of the numerical entities, so we group by the other values
        df_mea = df.groupby(hyperparameters).mean().reset_index()
        df_std = df.groupby(hyperparameters).std().reset_index()

        # We merge together the averaged results and the std-ed results
        df_mea_std = df_mea.merge(df_std, left_on=hyperparameters,
                                          right_on=hyperparameters)

        # We now take the best set of hyperparameters...
        df_best = df_mea[last_distinctions + [metrics[i]]].groupby(last_distinctions).min().reset_index()

        # ... and retrieve their corresponding averaged and std-ed error and other results.
        # The averaged results take the same name as the averaged ones, but with "_x" appended (std-ed: "_y").
        numerical_results_x, numerical_results_y = [], []
        for num in [metrics[i]] + numerical_results:
            numerical_results_x.append(num + '_x')
            numerical_results_y.append(num + '_y')

        df_best_with_info = df_best.merge(df_mea_std, left_on=last_distinctions+[metrics[i]],
                                                      right_on=last_distinctions+[metrics[i] + '_x'],
                                                      how='left')

        # We remove the doubloons
        df_best_with_info = df_best_with_info.groupby(last_distinctions).min().reset_index()

        # We make sure to only keep the relevant features
        df_best_with_info = df_best_with_info[last_distinctions + numerical_results_x + numerical_results_y]

        # We make sure the lines are properly sorted
        df_best_with_info.sort_values(by=last_distinctions + numerical_results_x + numerical_results_y)

        # We (optionaly) display the different columns of the resulting pandas dataframe.
        if show:
            for j in range(len(df_best_with_info.columns)):
                print(j, df_best_with_info.columns[j])
            print()

        # We update the relevant results dictinnary
        final_results[f"{i}"] = np.array(df_best_with_info)

    # Finally, we generate the latex table
    for pred_name in np.unique(final_results['0'][:, 0]):
        n_datasets = len(np.unique(final_results['0'][final_results['0'][:, 0] == pred_name, 1]))
        if type == 'regular':
            generate_latex_table(final_results, pred_name, n_datasets)
        elif type == 'error_min':
            generate_error_latex_table(final_results, pred_name, n_datasets)
        elif type == 'distributions':
            generate_distr_latex_table(final_results, pred_name, n_datasets)
        else:
            assert False

def generate_latex_table(final_results, pred_name, n_datasets):
    first = "\multirow{"+str(n_datasets)+"}{*}{"+pred_name+"}"
    last_dataset = ''
    Ben, FO, SO, Bin, CBound, Test = '', '', '', '', '', ''
    Ben_err, FO_err, SO_err, Bin_err, CBound_err, Test_err = '', '', '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std, CBound_std, Test_std = '', '', '', '', '', ''
    Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std, Test_err_std = '', '', '', '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] == pred_name:
            if final_results["0"][i][1] != last_dataset and last_dataset != '':
                #print(f"{first} & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} \\\ ")
                print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
                first = ''
                Ben, FO, SO, Bin, CBound, Test = '', '', '', '', '', ''
                Ben_err, FO_err, SO_err, Bin_err, CBound_err, Test_err = '', '', '', '', '', ''
                Ben_std, FO_std, SO_std, Bin_std, CBound_std, Test_std = '', '', '', '', '', ''
                Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std, Test_err_std = '', '', '', '', '', ''
            last_dataset = final_results["0"][i][1]
            if final_results["0"][i][2] == 'FO':
                FO = np.round(final_results["1"][i][3] * 100, 1)
                FO_err = np.round(final_results["1"][i][4] * 100, 1)
                FO_std = np.round(final_results["1"][i][6] * 100, 1)
                FO_err_std = np.round(final_results["1"][i][7] * 100, 1)
                Ben = np.round(final_results["0"][i][3] * 100, 1)
                Ben_err = np.round(final_results["0"][i][4] * 100, 1)
                Ben_std = np.round(final_results["0"][i][6] * 100, 1)
                Ben_err_std = np.round(final_results["0"][i][7] * 100, 1)
            elif final_results["0"][i][2] == 'SO':
                SO = np.round(final_results["1"][i][3] * 100, 1)
                SO_err = np.round(final_results["1"][i][4] * 100, 1)
                SO_std = np.round(final_results["1"][i][6] * 100, 1)
                SO_err_std = np.round(final_results["1"][i][7] * 100, 1)
            elif final_results["1"][i][2] == 'Bin':
                Bin = np.round(final_results["1"][i][3] * 100, 1)
                Bin_err = np.round(final_results["1"][i][4] * 100, 1)
                Bin_std = np.round(final_results["1"][i][6] * 100, 1)
                Bin_err_std = np.round(final_results["1"][i][7] * 100, 1)
            elif final_results["1"][i][2] == 'Cbound':
                CBound = np.round(final_results["1"][i][3] * 100, 1)
                CBound_err = np.round(final_results["1"][i][5] * 100, 1)
                CBound_std = np.round(final_results["1"][i][6] * 100, 1)
                CBound_err_std = np.round(final_results["1"][i][8] * 100, 1)
            elif final_results["0"][i][2] == 'VCdim':
                Test = np.round(final_results["1"][i][3] * 100, 1)
                Test_err = np.round(final_results["1"][i][4] * 100, 1)
                Test_std = np.round(final_results["1"][i][6] * 100, 1)
                Test_err_std = np.round(final_results["1"][i][7] * 100, 1)
    # print(f"{first} & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} \\\ ")
    print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
    print("\hline")

def generate_error_latex_table(final_results, pred_name, n_datasets):
    first = "\multirow{"+str(n_datasets)+"}{*}{"+pred_name+"}"
    last_dataset = ''
    Ben, FO, SO, Bin, CBound, Test = '', '', '', '', '', ''
    Ben_err, FO_err, SO_err, Bin_err, CBound_err, Test_err = '', '', '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std, CBound_std, Test_std = '', '', '', '', '', ''
    Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std, Test_err_std = '', '', '', '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] == pred_name:
            if final_results["0"][i][1] != last_dataset and last_dataset != '':
                #print(f"{first} & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} \\\ ")
                print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
                first = ''
                Ben, FO, SO, Bin, CBound, Test = '', '', '', '', '', ''
                Ben_err, FO_err, SO_err, Bin_err, CBound_err, Test_err = '', '', '', '', '', ''
                Ben_std, FO_std, SO_std, Bin_std, CBound_std, Test_std = '', '', '', '', '', ''
                Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std, Test_err_std = '', '', '', '', '', ''
            last_dataset = final_results["0"][i][1]
            if final_results["0"][i][2] == 'FO':
                FO = np.round(final_results["1"][i][3] * 100, 1)
                FO_err = np.round(final_results["1"][i][4] * 100, 1)
                FO_std = np.round(final_results["1"][i][6] * 100, 1)
                FO_err_std = np.round(final_results["1"][i][7] * 100, 1)
                Ben = np.round(final_results["0"][i][5] * 100, 1)
                Ben_err = np.round(final_results["0"][i][4] * 100, 1)
                Ben_std = np.round(final_results["0"][i][8] * 100, 1)
                Ben_err_std = np.round(final_results["0"][i][7] * 100, 1)
            elif final_results["0"][i][2] == 'SO':
                SO = np.round(final_results["1"][i][3] * 100, 1)
                SO_err = np.round(final_results["1"][i][4] * 100, 1)
                SO_std = np.round(final_results["1"][i][6] * 100, 1)
                SO_err_std = np.round(final_results["1"][i][7] * 100, 1)
            elif final_results["1"][i][2] == 'Bin':
                Bin = np.round(final_results["1"][i][3] * 100, 1)
                Bin_err = np.round(final_results["1"][i][4] * 100, 1)
                Bin_std = np.round(final_results["1"][i][6] * 100, 1)
                Bin_err_std = np.round(final_results["1"][i][7] * 100, 1)
            elif final_results["1"][i][2] == 'Cbound':
                CBound = np.round(final_results["1"][i][3] * 100, 1)
                CBound_err = np.round(final_results["1"][i][4] * 100, 1)
                CBound_std = np.round(final_results["1"][i][6] * 100, 1)
                CBound_err_std = np.round(final_results["1"][i][7] * 100, 1)
            elif final_results["0"][i][2] == 'Test':
                Test = np.round(final_results["1"][i][3] * 100, 1)
                Test_err = np.round(final_results["1"][i][4] * 100, 1)
                Test_std = np.round(final_results["1"][i][6] * 100, 1)
                Test_err_std = np.round(final_results["1"][i][7] * 100, 1)
    # print(f"{first} & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} \\\ ")
    print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Test} $\pm$ {Test_std} & {Test_err} $\pm$ {Test_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std}\\\ ")
    print("\hline")

def generate_distr_latex_table(final_results, pred_name, n_datasets):
    first = "\multirow{"+str(n_datasets)+"}{*}{"+pred_name+"}"
    last_dataset = ''
    categorical, dirichlet, gaussian = '', '', ''
    categorical_err, dirichlet_err, gaussian_err = '', '', ''
    categorical_std, dirichlet_std, gaussian_std = '', '', ''
    categorical_err_std, dirichlet_err_std, gaussian_err_std = '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] == pred_name:
            if final_results["0"][i][1] != last_dataset and last_dataset != '':
                print(f"{first} & {last_dataset} & {categorical} $\pm$ {categorical_std} & {categorical_err} $\pm$ {categorical_err_std} & {dirichlet} $\pm$ {dirichlet_std} & {dirichlet_err} $\pm$ {dirichlet_err_std} & {gaussian} $\pm$ {gaussian_std} & {gaussian_err} $\pm$ {gaussian_err_std} \\\ ")
                first = ''
                categorical, dirichlet, gaussian = '', '', ''
                categorical_err, dirichlet_err, gaussian_err = '', '', ''
                categorical_std, dirichlet_std, gaussian_std = '', '', ''
                categorical_err_std, dirichlet_err_std, gaussian_err_std = '', '', ''
            last_dataset = final_results["0"][i][1]
            if final_results["0"][i][2] == 'categorical':
                categorical = np.round(final_results["0"][i][3] * 100, 1)
                categorical_err = np.round(final_results["0"][i][4] * 100, 1)
                categorical_std = np.round(final_results["0"][i][6] * 100, 1)
                categorical_err_std = np.round(final_results["0"][i][7] * 100, 1)
            elif final_results["0"][i][2] == 'dirichlet':
                dirichlet = np.round(final_results["0"][i][3] * 100, 1)
                dirichlet_err = np.round(final_results["0"][i][4] * 100, 1)
                dirichlet_std = np.round(final_results["0"][i][6] * 100, 1)
                dirichlet_err_std = np.round(final_results["0"][i][7] * 100, 1)
            elif final_results["0"][i][2] == 'gaussian':
                gaussian = np.round(final_results["0"][i][3] * 100, 1)
                gaussian_err = np.round(final_results["0"][i][4] * 100, 1)
                gaussian_std = np.round(final_results["0"][i][6] * 100, 1)
                gaussian_err_std = np.round(final_results["0"][i][7] * 100, 1)
    print(f"{first} & {last_dataset} & {categorical} $\pm$ {categorical_std} & {categorical_err} $\pm$ {categorical_err_std} & {dirichlet} $\pm$ {dirichlet_std} & {dirichlet_err} $\pm$ {dirichlet_err_std} & {gaussian} $\pm$ {gaussian_std} & {gaussian_err} $\pm$ {gaussian_err_std} \\\ ")
    print("\hline")

file = 'wandb_export.csv'
hyperparams = ['M', 'batch_size', 'dataset', 'delta', 'distribution', 'is_using_wandb', 'lr', 'num_epochs',
               'num_trials', 'num_workers', 'order', 'pred', 'prior', 'project_name', 'rand_N', 'risk', 'stump_init']
csv_to_latex(file_name=file,
             hyperparameters=hyperparams,
             type='regular',  # type \in ['regular', 'error_min', 'granular', 'distributions']
             show=False)