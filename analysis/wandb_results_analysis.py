import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def csv_to_latex(file_name, hyperparameters, numerical_results, metrics, last_distinctions, show=False, granul=False):
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
    for pred_name in np.unique(final_results['0'][:, 0]):
        n_datasets = len(np.unique(final_results['0'][final_results['0'][:, 0] == pred_name, 1]))
        if granul:
            generate_granul_latex_table(final_results, pred_name, n_datasets)
        else:
            generate_latex_table(final_results, pred_name, n_datasets)

def generate_latex_table(final_results, pred_name, n_datasets):
    first = "\multirow{"+str(n_datasets)+"}{*}{"+pred_name+"}"
    last_dataset = ''
    Ben, FO, SO, Bin, CBound = '', '', '', '', ''
    Ben_err, FO_err, SO_err, Bin_err, CBound_err = '', '', '', '', ''
    Ben_std, FO_std, SO_std, Bin_std, CBound_std = '', '', '', '', ''
    Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std = '', '', '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] == pred_name:
            if final_results["0"][i][1] != last_dataset and last_dataset != '':
                print(f"{first} & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
                first = ''
                Ben, FO, SO, Bin, CBound = '', '', '', '', ''
                Ben_err, FO_err, SO_err, Bin_err, CBound_err = '', '', '', '', ''
                Ben_std, FO_std, SO_std, Bin_std, CBound_std = '', '', '', '', ''
                Ben_err_std, FO_err_std, SO_err_std, Bin_err_std, CBound_err_std = '', '', '', '', ''
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
    print(f" & {last_dataset} & {FO} $\pm$ {FO_std} & {FO_err} $\pm$ {FO_err_std} & {SO} $\pm$ {SO_std} & {SO_err} $\pm$ {SO_err_std} & {Bin} $\pm$ {Bin_std} & {Bin_err} $\pm$ {Bin_err_std} & {CBound} $\pm$ {CBound_std} & {CBound_err} $\pm$ {CBound_err_std} & {Ben} $\pm$ {Ben_std} & {Ben_err} $\pm$ {Ben_err_std} \\\ ")
    print("\hline")

def generate_granul_latex_table(final_results, pred_name, n_datasets):
    first = "\multirow{"+str(n_datasets)+"}{*}{"+pred_name+"}"
    last_dataset = ''
    Triple, Part, Both = '', '', ''
    Triple_err, Part_err, Both_err = '', '', ''
    Triple_std, Part_std, Both_std = '', '', ''
    Triple_err_std, Part_err_std, Both_err_std = '', '', ''
    for i in range(len(final_results["0"])):
        if final_results["0"][i][0] == pred_name:
            if final_results["0"][i][1] != last_dataset and last_dataset != '':
                print(f"{first} & {last_dataset} & {Triple} $\pm$ {Triple_std} & {Triple_err} $\pm$ {Triple_err_std} & {Part} $\pm$ {Part_std} & {Part_err} $\pm$ {Part_err_std} & {Both} $\pm$ {Both_std} & {Both_err} $\pm$ {Both_err_std} \\\ ")
                first = ''
                Triple, Part, Both = '', '', ''
                Triple_err, Part_err, Both_err = '', '', ''
                Triple_std, Part_std, Both_std = '', '', ''
                Triple_err_std, Part_err_std, Both_err_std = '', '', ''
            last_dataset = final_results["0"][i][1]
            if final_results["0"][i][2] == 'FO':
                Part = np.round(final_results["0"][i][3] * 100, 1)
                Part_err = np.round(final_results["0"][i][4] * 100, 1)
                Part_std = np.round(final_results["0"][i][6] * 100, 1)
                Part_err_std = np.round(final_results["0"][i][7] * 100, 1)
                Triple = np.round(final_results["1"][i][3] * 100, 1)
                Triple_err = np.round(final_results["1"][i][4] * 100, 1)
                Triple_std = np.round(final_results["1"][i][6] * 100, 1)
                Triple_err_std = np.round(final_results["1"][i][7] * 100, 1)
                Both = np.round(final_results["2"][i][3] * 100, 1)
                Both_err = np.round(final_results["2"][i][4] * 100, 1)
                Both_std = np.round(final_results["2"][i][6] * 100, 1)
                Both_err_std = np.round(final_results["2"][i][7] * 100, 1)
    print(f"{first} & {last_dataset} & {Triple} $\pm$ {Triple_std} & {Triple_err} $\pm$ {Triple_err_std} & {Part} $\pm$ {Part_std} & {Part_err} $\pm$ {Part_err_std} & {Both} $\pm$ {Both_std} & {Both_err} $\pm$ {Both_err_std} \\\ ")
    print("\hline")

file = 'wandb_export_2025-09-04T12_33_42.940-04_00.csv'
hyperparams = ['M', 'batch_size', 'dataset', 'delta', 'distribution', 'is_using_wandb', 'lr', 'num_epochs',
               'num_trials', 'num_workers', 'order', 'pred', 'prior', 'project_name', 'rand_n', 'risk', 'stump_init']
num_results = ['test-error_finetune', 'test-error']  # Do not change that order
metric = ['part_triple_bnd_tnd', 'deterministic_bound']
#metric = ['part_bnd_tnd', 'triple_bnd_tnd', 'part_triple_bnd_tnd']
final_distinction = ['pred', 'dataset', 'risk']   # Do not change that order
csv_to_latex(file_name=file,
             hyperparameters=hyperparams,
             numerical_results=num_results,
             metrics=metric,
             last_distinctions=final_distinction,
             show=False,
             granul=False)