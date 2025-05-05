def create_config_dico(cfg):
    """
    Given a multi-level dictionary of config, returns a single-level dictionary of config.
    """
    if type(cfg['dataset']) == str:
        dataset_name = cfg['dataset']
    else:
        dataset_name = cfg['dataset']['distr']
    dico = {'project_name': cfg['project_name'],
            'num_trials': cfg['num_trials'],
            'num_workers': cfg['num_workers'],
            'is_using_wandb': cfg['is_using_wandb'],
            'M': cfg['model']['M'],
            'prior': cfg['model']['prior'],
            'pred': cfg['model']['pred'],
            'bootstrap': cfg['model']['bootstrap'],
            'tree_depth': cfg['model']['tree_depth'],
            'm': cfg['model']['m'],
            'uniform': cfg['model']['uniform'],
            'seed': cfg['training']['seed'],
            'lr': cfg['training']['lr'],
            'batch_size': cfg['training']['batch_size'],
            'num_epochs': cfg['training']['num_epochs'],
            'risk': cfg['training']['risk'],
            'distribution': cfg['training']['distribution'],
            'opt_bound': cfg['training']['opt_bound'],
            'sigmoid_c': cfg['training']['sigmoid_c'],
            'rand_n': cfg['training']['rand_n'],
            'MC_draws': cfg['training']['MC_draws'],
            'dataset': dataset_name,
            'delta': cfg['bound']['delta'],
            'type':  cfg['bound']['type'],
            'stochastic':  cfg['bound']['stochastic']}
    return dico

def create_run_name(config):
    """
    Given a dictionary of config, returns a run name.
    """
    return 'dataset=' + config['dataset'] + '_distr=' + config['distribution'] + '_risk=' + config['risk'] + '_pred-type=' + \
           config['pred'] + '_M=' + str(config['M']) + '_prior=' + str(config['prior']) + '_seed=' + str(config['seed'])