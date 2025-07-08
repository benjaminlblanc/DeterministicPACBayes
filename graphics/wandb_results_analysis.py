import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def csv_to_latex(file_name, hyperparameters, numerical_results, metrics, last_distinctions, show=False):
    """
    Generates latex table from csv file.
    """
    # We first import the data
    try:
        with open(file_name, 'r') as det_ :
            det = [line.replace('"', '').replace('\n', '').split(',') for line in det_]
    except FileNotFoundError:
        assert False, ("Make sure to download a WandB results csv file, to put it in the 'graphics' folder, and to call"
                       " the correct name.")
    det_.close()
    initial_df = pd.DataFrame(det[1:], columns=det[0])

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
        df_best_with_info = df_best_with_info[final_distinction + numerical_results_x + numerical_results_y]

        # We make sure the lines are properly sorted
        df_best_with_info.sort_values(by=final_distinction + numerical_results_x + numerical_results_y)

        # We (optionaly) display the different columns of the resulting pandas dataframe.
        if show:
            for j in range(len(df_best_with_info.columns)):
                print(j, df_best_with_info.columns[j])
            print()

        # We update the relevant results dictinnary
        final_results[f"{i}"] = np.array(df_best_with_info)

    # Finally, we generate the latex table
    generate_latex_table(final_results)

def generate_latex_table(final_results):
    last_dataset = ''
    Ben, FO, SO, Bin, Dis_Renyi, Dis_KL = '', '', '', '', '', ''
    Ben_err, FO_err, SO_err, Bin_err, Dis_Renyi_err, Dis_KL_err = '', '', '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std, Dis_Renyi_std, Dis_KL_std = '', '', '', '', '', ''
    Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, Dis_Renyi_err_std, Dis_KL_err_std = '', '', '', '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] != last_dataset and last_dataset != '':
            print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {Dis_Renyi} $\pm$ {Dis_Renyi_std} & {Dis_Renyi_err} $\pm$ {Dis_Renyi_err_std} & {Dis_KL} $\pm$ {Dis_KL_std} & {Dis_KL_err} $\pm$ {Dis_KL_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
            Ben, FO, SO, Bin, Dis_Renyi, Dis_KL = '', '', '', '', '', ''
            Ben_err, FO_err, SO_err, Bin_err, Dis_Renyi_err, Dis_KL_err = '', '', '', '', '', ''
            Ben_std, FO_std, SO_std, Bin_std, Dis_Renyi_std, Dis_KL_std = '', '', '', '', '', ''
            Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, Dis_Renyi_err_std, Dis_KL_err_std = '', '', '', '', '', ''
        last_dataset = final_results["0"][i][0]
        if final_results["0"][i][1] == 'FO':
            FO = np.round(final_results["1"][i][2] * 100, 1)
            FO_std = np.round(final_results["1"][i][8] * 100, 1)
            FO_err = np.round(final_results["1"][i][4] * 100, 1)
            FO_err_std = np.round(final_results["1"][i][10] * 100, 1)
            Ben = np.round(final_results["0"][i][2] * 100, 1)
            Ben_std = np.round(final_results["0"][i][8] * 100, 1)
            Ben_err = np.round(final_results["0"][i][4] * 100, 1)
            Ben_err_std = np.round(final_results["0"][i][10] * 100, 1)
            Dis_KL = np.round(final_results["2"][i][2] * 100, 1)
            Dis_KL_std = np.round(final_results["2"][i][3] * 100, 1)
            Dis_KL_err = np.round(final_results["2"][i][6] * 100, 1)
            Dis_KL_err_std = np.round(final_results["2"][i][7] * 100, 1)
        elif final_results["0"][i][1] == 'SO':
            SO = np.round(final_results["1"][i][2] * 100, 1)
            SO_std = np.round(final_results["1"][i][8] * 100, 1)
            SO_err = np.round(final_results["1"][i][4] * 100, 1)
            SO_err_std = np.round(final_results["1"][i][10] * 100, 1)
        elif final_results["1"][i][1] == 'Bin':
            Bin = np.round(final_results["1"][i][2] * 100, 1)
            Bin_std = np.round(final_results["1"][i][8] * 100, 1)
            Bin_err = np.round(final_results["1"][i][4] * 100, 1)
            Bin_err_std = np.round(final_results["1"][i][10] * 100, 1)
        elif final_results["1"][i][1] == 'Dis_Renyi':
            Dis_Renyi = np.round(final_results["2"][i][2] * 100, 1)
            Dis_Renyi_std = np.round(final_results["2"][i][3] * 100, 1)
            Dis_Renyi_err = np.round(final_results["2"][i][6] * 100, 1)
            Dis_Renyi_err_std = np.round(final_results["2"][i][7] * 100, 1)
    print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {Dis_Renyi} $\pm$ {Dis_Renyi_std} & {Dis_Renyi_err} $\pm$ {Dis_Renyi_err_std} & {Dis_KL} $\pm$ {Dis_KL_std} & {Dis_KL_err} $\pm$ {Dis_KL_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")

file = 'wandb_export_2025-07-07T10_29_29.192-04_00.csv'
hyperparams = ['M', 'batch_size', 'bootstrap', 'dataset', 'delta', 'distribution', 'is_using_wandb', 'lr',
               'num_epochs', 'num_trials', 'num_workers', 'opt_bound', 'order', 'pred', 'prior', 'project_name', 'rand_n',
               'risk', 'stochastic', 'stump_init', 'type', 'uniform']
num_results = ['deterministic_bound_sampled_std', 'test-error', 'test-error_finetune', 'test-error_sampled', 'test-error_sampled_std']  # Do not change that order
metric = ['ben_bound_with_finetune', 'deterministic_bound', 'deterministic_bound_sampled']
final_distinction = ['dataset', 'risk']   # Do not change that order
csv_to_latex(file_name=file,
             hyperparameters=hyperparams,
             numerical_results=num_results,
             metrics=metric,
             last_distinctions=final_distinction,
             show=False)
