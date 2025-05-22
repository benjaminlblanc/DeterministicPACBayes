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
    Ben, FO, SO, Bin, Dis_Renyi, Dis_KL = '', '', '', '', '', ''
    Ben_err, FO_err, SO_err, Bin_err, Dis_Renyi_err, Dis_KL_err = '', '', '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std, Dis_Renyi_std, Dis_KL_std = '', '', '', '', '', ''
    Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, Dis_Renyi_err_std, Dis_KL_err_std = '', '', '', '', '', ''
    for i in range(len(results)):
        if results[i][0] != last_dataset and last_dataset != '':
            print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {Dis_Renyi} $\pm$ {Dis_Renyi_std} & {Dis_Renyi_err} $\pm$ {Dis_Renyi_err_std} & {Dis_KL} $\pm$ {Dis_KL_std} & {Dis_KL_err} $\pm$ {Dis_KL_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
            Ben, FO, SO, Bin, Dis_Renyi, Dis_KL = '', '', '', '', '', ''
            Ben_err, FO_err, SO_err, Bin_err, Dis_Renyi_err, Dis_KL_err = '', '', '', '', '', ''
            Ben_std, FO_std, SO_std, Bin_std, Dis_Renyi_std, Dis_KL_std = '', '', '', '', '', ''
            Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, Dis_Renyi_err_std, Dis_KL_err_std = '', '', '', '', '', ''
        last_dataset = results[i][0]
        if results[i][1] == 'FO':
            FO = np.round(results[i][3], 3)
            FO_std = np.round(results[i][7], 3)
            FO_err = np.round(results[i][4], 3)
            FO_err_std = np.round(results[i][8], 3)
            Ben = np.round(results[i][2], 3)
            Ben_std = np.round(results[i][6], 3)
            Ben_err = np.round(results[i][4], 3)
            Ben_err_std = np.round(results[i][8], 3)
        elif results[i][1] == 'SO':
            SO = np.round(results[i][3], 3)
            SO_std = np.round(results[i][7], 3)
            SO_err = np.round(results[i][4], 3)
            SO_err_std = np.round(results[i][8], 3)
        elif results[i][1] == 'Bin':
            Bin = np.round(results[i][3], 3)
            Bin_std = np.round(results[i][7], 3)
            Bin_err = np.round(results[i][4], 3)
            Bin_err_std = np.round(results[i][8], 3)
        elif results[i][1] == 'Dis_Renyi':
            Dis_Renyi = np.round(results[i][3], 3)
            Dis_Renyi_std = np.round(results[i][7], 3)
            Dis_Renyi_err = np.round(results[i][4], 3)
            Dis_Renyi_err_std = np.round(results[i][5], 3)
        elif results[i][1] == 'Dis_KL':
            Dis_KL = np.round(results[i][3], 3)
            Dis_KL_std = np.round(results[i][7], 3)
            Dis_KL_err = np.round(results[i][4], 3)
            Dis_KL_err_std = np.round(results[i][5], 3)
    print(f"{last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {Dis_Renyi} $\pm$ {Dis_Renyi_std} & {Dis_Renyi_err} $\pm$ {Dis_Renyi_err_std} & {Dis_KL} $\pm$ {Dis_KL_std} & {Dis_KL_err} $\pm$ {Dis_KL_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")

file = 'wandb_export_2025-05-22T09_49_29.224-04_00.csv'
hyperparams = ['M', 'batch_size', 'bootstrap', 'dataset', 'delta', 'distribution', 'is_using_wandb', 'lr',
               'num_epochs', 'num_trials', 'num_workers', 'opt_bound', 'pred', 'prior', 'project_name', 'rand_n',
               'risk', 'seed', 'stochastic', 'stump_init', 'tree_depth', 'type', 'uniform']
num_results = ['ben_bound_with_finetune', 'deterministic_bound', 'test-error_finetune', 'test-error_std_finetune']  # Do not change that order
final_distinction = ['dataset', 'risk']   # Do not change that order
csv_to_latex(file_name=file,
             hyperparameters=hyperparams,
             numerical_results=num_results,
             last_distinctions=final_distinction)

#ADULT & 0.47 $\pm$ 0.001 & 0.222 $\pm$ 0.001 & 0.534 $\pm$ 0.001 & 0.17 $\pm$ 0.003 & 0.366 $\pm$ 0.001 & 0.172 $\pm$ 0.002 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.439 $\pm$ 0.016 & 0.222 $\pm$ 0.001 \\
#CODRNA & 0.302 $\pm$ 0.004 & 0.119 $\pm$ 0.002 & 0.519 $\pm$ 0.005 & 0.124 $\pm$ 0.002 & 0.31 $\pm$ 0.002 & 0.25 $\pm$ 0.001 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.151 $\pm$ 0.002 & 0.119 $\pm$ 0.002 \\
#HABER & 0.753 $\pm$ 0.01 & 0.265 $\pm$ 0.018 & 1.142 $\pm$ 0.017 & 0.255 $\pm$ 0.029 & 0.811 $\pm$ 0.013 & 0.252 $\pm$ 0.033 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.379 $\pm$ 0.005 & 0.265 $\pm$ 0.018 \\
#MUSH & 0.097 $\pm$ 0.002 & 0.014 $\pm$ 0.001 & 0.244 $\pm$ 0.006 & 0.016 $\pm$ 0.002 & 0.205 $\pm$ 0.012 & 0.186 $\pm$ 0.004 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.048 $\pm$ 0.001 & 0.014 $\pm$ 0.001 \\
#PHIS & 0.258 $\pm$ 0.003 & 0.082 $\pm$ 0.006 & 0.385 $\pm$ 0.002 & 0.085 $\pm$ 0.005 & 0.258 $\pm$ 0.003 & 0.107 $\pm$ 0.006 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.252 $\pm$ 0.01 & 0.082 $\pm$ 0.006 \\
#SVMGUIDE & 0.175 $\pm$ 0.006 & 0.058 $\pm$ 0.01 & 0.286 $\pm$ 0.007 & 0.07 $\pm$ 0.01 & 0.196 $\pm$ 0.005 & 0.072 $\pm$ 0.009 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.087 $\pm$ 0.003 & 0.058 $\pm$ 0.01 \\
#TTT & 0.846 $\pm$ 0.016 & 0.318 $\pm$ 0.022 & 1.08 $\pm$ 0.049 & 0.358 $\pm$ 0.008 & 0.839 $\pm$ 0.048 & 0.398 $\pm$ 0.008 &  $\pm$  &  $\pm$  &  $\pm$  &  $\pm$  & 0.473 $\pm$ 0.01 & 0.318 $\pm$ 0.022 \\