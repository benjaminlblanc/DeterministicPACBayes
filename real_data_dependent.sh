name=DeterministicPACBayes
wandb=True
prd=rf
m=100
trials=5
seed=541944

for d in SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION
do
  for r in FO
  do
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=1 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones training.seed=$seed
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones training.seed=$seed
    #python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.seed=$seed
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.seed=$seed
	done

	for r in SO
  do
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones training.seed=$seed
    #python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.seed=$seed
    python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.seed=$seed
	done

	for r in Bin
  do
    for r_n in 10 100
    do
      python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=ones training.rand_n=$r_n training.seed=$seed
      #python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.rand_n=$r_n training.seed=$seed
      python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand training.rand_n=$r_n training.seed=$seed
    done
	done

	for r in Dis_Renyi
  do
    #for ordr in 1.05 1.2 1.5 1.8 1.95
    #do
    #  python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand bound.order=$ordr bound.n_grid=5 training.seed=$seed
    #done
    for ordr in 1.05 1.5 2 5 10
    do
      python3 real.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb model.stump_init=rand bound.order=$ordr bound.n_grid=5 training.seed=$seed
    done
	done
done