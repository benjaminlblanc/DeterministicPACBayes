# A Framework for Bounding Deterministic Risk with PAC-Bayes: Applications to Majority Votes

A work by [Leblanc and Germain](https://arxiv.org/abs/2510.25569).

### Dependencies

- Use Python 3.8.20.
- Install the other requirements from the `requirements.txt`

### Reproducing the main results

To reproduce the paper's main results, run `bash run/run_stumps.sh` and `bash run/run_forests.sh` in the terminal.

### Saving the runs in Weights & Biases

Set the `is_using_wandb` hyperparameter from the shell script that is used (`run/run_stumps.sh` or `run/run_forests.sh`) to True. The default value is True.

### Managing the other hyperparameters

The generic hyperparameters are found in `base_config.yaml`. The more specific hyperparameters are found in the shell scripts. Take a look at the `hyperparameters_dictionary.txt` file to learn more about the effect of the many tunable hyperparameters.

### Shoutout to some inspiration

- A general adaptation of the code from [Learning Stochastic Majority Votes by Minimizing a PAC-Bayes Generalization Bound](https://github.com/vzantedeschi/StocMV). 
- Inspiration from [rnoxy](https://github.com/rnoxy/cifar10-cnn) for creating the embedded datasets.
- Adaptation of [Self-Bounding Majority Vote Learning Algorithms by the Direct Minimization of a Tight PAC-Bayesian C-Bound](https://github.com/paulviallard/ECML21-PB-CBound) for the C-Bound algorithm.
