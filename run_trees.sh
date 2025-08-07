name=DeterministicPACBayes
wandb=True
prd=rf
m=100
trials=5
seed=541944

for d in FASHION MNIST PENDIGITS PROTEIN SHUTTLE
do
  r=FO
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=dirichlet model.prior=1 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed

  r=SO
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed

  r=Bin
  for r_n in 10 100
  do
    python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.rand_n=$r_n training.seed=$seed
  done
done