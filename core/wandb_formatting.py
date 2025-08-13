def create_config_dico(cfg):
    """
    Given a multi-level dictionary of config, returns a single-level dictionary of config.
    """
    if type(cfg['dataset']) == str:
        dataset_name = cfg['dataset']
    else:
        dataset_name = cfg['dataset']['distr']
    dico = {'project_name': cfg['project_name'],
            'dataset': dataset_name,
            'num_trials': cfg['num_trials'],
            'num_workers': cfg['num_workers'],
            'is_using_wandb': cfg['is_using_wandb'],
            'M': cfg['model']['M'],
            'prior': cfg['model']['prior'],
            'pred': cfg['model']['pred'],
            'stump_init': cfg['model']['stump_init'],
            'samples_prop': cfg['model']['samples_prop'],
            'max_tree_depth': cfg['model']['max_tree_depth'],
            'posterior_std': cfg['model']['posterior_std'],
            'output': cfg['model']['output'],
            'seed': cfg['training']['seed'],
            'lr': cfg['training']['lr'],
            'batch_size': cfg['training']['batch_size'],
            'normalize_data': cfg['training']['normalize_data'],
            'num_epochs': cfg['training']['num_epochs'],
            'risk': cfg['training']['risk'],
            'distribution': cfg['training']['distribution'],
            'rand_n': cfg['training']['rand_n'],
            'delta': cfg['bound']['delta'],
            'type':  cfg['bound']['type'],
            'order':  cfg['bound']['order']}
    return dico

def create_run_name(config, seed):
    """
    Given a dictionary of config, returns a run name.
    """
    return 'dataset=' + config['dataset'] + '_distr=' + config['distribution'] + '_risk=' + config['risk'] + '_pred-type=' + \
           config['pred'] + '_stump-init=' + str(config['stump_init']) + '_output=' + str(config['output']) + '_prior=' + str(config['prior']) + '_seed=' + str(seed)