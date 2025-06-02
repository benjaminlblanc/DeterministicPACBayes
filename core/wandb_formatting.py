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
            'stump_init': cfg['model']['stump_init'],
            'bootstrap': cfg['model']['bootstrap'],
            'tree_depth': cfg['model']['tree_depth'],
            'uniform': cfg['model']['uniform'],
            'seed': cfg['training']['seed'],
            'lr': cfg['training']['lr'],
            'batch_size': cfg['training']['batch_size'],
            'num_epochs': cfg['training']['num_epochs'],
            'risk': cfg['training']['risk'],
            'distribution': cfg['training']['distribution'],
            'opt_bound': cfg['training']['opt_bound'],
            'rand_n': cfg['training']['rand_n'],
            'dataset': dataset_name,
            'delta': cfg['bound']['delta'],
            'type':  cfg['bound']['type'],
            'stochastic':  cfg['bound']['stochastic'],
            'order':  cfg['bound']['order']}
    return dico

def create_run_name(config, seed):
    """
    Given a dictionary of config, returns a run name.
    """
    return 'dataset=' + config['dataset'] + '_distr=' + config['distribution'] + '_risk=' + config['risk'] + '_pred-type=' + \
           config['pred'] + '_stump-init=' + str(config['stump_init']) + '_M=' + str(config['M']) + '_prior=' + str(config['prior']) + '_seed=' + str(seed)