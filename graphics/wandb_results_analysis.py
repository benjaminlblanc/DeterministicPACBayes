import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def csv_to_latex(file_name, hyperparameters, numerical_results, last_distinctions):
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
    df = pd.DataFrame(det[1:], columns=det[0])

    # We only keep the relevant column values
    df = df[hyperparameters + numerical_results]

    # We transform the numerical entities to floats from strings
    for numerical in numerical_results:
        df = df.astype({numerical: float})

    # We create the average and std of the numerical entities, so we group by the other values
    df_mea = df.groupby(hyperparameters).mean().reset_index()
    df_std = df.groupby(hyperparameters).std().reset_index()

    # We merge together the averaged results and the std-ed results
    df_mea_std = df_mea.merge(df_std, left_on=hyperparameters,
                                      right_on=hyperparameters)

    # We now take the best set of hyperparameters...
    df_best = df_mea[last_distinctions + numerical_results].groupby(last_distinctions).min().reset_index()

    # ... and retrieve their std-ed results
    # The std-ed results take the same name as the averaged ones, but with "_x" appended
    numerical_results_x, numerical_results_y = [], []
    for num in numerical_results:
        numerical_results_x.append(num + '_x')
        numerical_results_y.append(num + '_y')
    df_best_with_info = df_best.merge(df_mea_std, left_on=last_distinctions+[numerical_results[1]],
                                                  right_on=last_distinctions+[numerical_results_x[1]],
                                                  how='left')

    # We make sure to only keep the relevant features
    df_best_with_info = df_best_with_info[final_distinction + numerical_results + numerical_results_y]

    # We make sure the lines are properly sorted
    df_best_with_info.sort_values(by=final_distinction + numerical_results + numerical_results_y)

    # Finally, we generate the latex table
    generate_latex_table(np.array(df_best_with_info))

def generate_latex_table(results):
    last_dataset = ''
    Ben, FO, SO, Bin = '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std = '', '', '', ''
    for i in range(len(results)):
        if results[i][0] != last_dataset and last_dataset != '':
            print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {SO} $\pm$ {SO_std} & {Bin} $\pm$ {Bin_std} & {Ben} $\pm$ "
                  f"{Ben_std}\\\ ")
            Ben, FO, SO, Bin = '', '', '', ''
            Ben_std, FO_std, SO_std, Bin_std = '', '', '', ''
        last_dataset = results[i][0]
        if results[i][1] == 'FO':
            FO = np.round(results[i][3], 3)
            FO_std = np.round(results[i][5], 3)
            Ben = np.round(results[i][2], 3)
            Ben_std = np.round(results[i][4], 3)
        elif results[i][1] == 'SO':
            SO = np.round(results[i][3], 3)
            SO_std = np.round(results[i][5], 3)
        elif results[i][1] == 'Bin':
            Bin = np.round(results[i][3], 3)
            Bin_std = np.round(results[i][5], 3)
    print(
        f"{last_dataset} & {FO} $\pm$ {FO_std} & {SO} $\pm$ {SO_std} & {Bin} $\pm$ {Bin_std} & {Ben} $\pm$ {Ben_std}\\\ ")

file = 'wandb_export_2025-05-21T17_17_50.339-04_00.csv'
hyperparams = ['M', 'batch_size', 'bootstrap', 'dataset', 'delta', 'distribution', 'is_using_wandb', 'lr',
               'num_epochs', 'num_trials', 'num_workers', 'opt_bound', 'pred', 'prior', 'project_name', 'rand_n',
               'risk', 'seed', 'stochastic', 'stump_init', 'tree_depth', 'type', 'uniform']
num_results = ['ben_bound_with_finetune', 'deterministic_bound']  # Do not change that order
final_distinction = ['dataset', 'risk']   # Do not change that order
csv_to_latex(file_name=file,
             hyperparameters=hyperparams,
             numerical_results=num_results,
             last_distinctions=final_distinction)