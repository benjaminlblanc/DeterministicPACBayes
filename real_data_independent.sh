name=Test
wandb=True
prd=stumps-uniform
m=10
trials=1

for d in MUSH SVMGUIDE HABER TTT CODRNA ADULT PHIS
do
  for r in FO
  do
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=1 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand
	done

	for r in SO
  do
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand
	done

	for r in Bin
  do
    for r_n in 10
    do
      python3 project_name=$name real.py num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones training.rand_n=$r_n
      python3 project_name=$name real.py num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.rand_n=$r_n
      python3 project_name=$name real.py num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.rand_n=$r_n
    done
	done

	for r in Dis_Renyi
  do
    for ordr in 1.05
    do
      python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand bound.order=$ordr bound.n_grid=5
    done
    for ordr in 1.05
    do
      python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand bound.order=$ordr bound.n_grid=5
    done
	done
done