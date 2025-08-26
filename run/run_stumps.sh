name=DeterministicPACBayes
wandb=True
prd=UniformStumps
m=10
trials=5
seed=541944

for d in ADULT CODRNA MUSH HABER PHIS SVMGUIDE TTT
do
  r=FO
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=dirichlet model.prior=1 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=dirichlet model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed

  r=SO
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed

  r=Bin
  for r_n in 10 100
  do
    python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.rand_n=$r_n training.seed=$seed
    python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.rand_n=$r_n training.seed=$seed
  done

  #r=Dis_Renyi
  #n_grd=5
  #for ordr in 1.05 1.5 2 5 10
  #do
  #  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb bound.order=$ordr bound.n_grid=$n_grd training.seed=$seed model.output=class
  #done
  #for ordr in 1.05 1.2 1.5 1.8 1.95
  #do
  #  python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb bound.order=$ordr bound.n_grid=$n_grd training.seed=$seed model.output=class
  #done

  r=Cbound
    python3 training.py project_name=$name num_trials=$trials dataset=$d model.M=$m training.lr=0.1 training.distribution=categorical model.prior=adjusted training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
done