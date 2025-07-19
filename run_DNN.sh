name=DeterministicPACBayes
wandb=True
prd=resnet152
trials=5
seed=541944

for d in MNIST
do
  r=FO
  python3 training.py project_name=$name num_trials=$trials dataset=$d training.lr=0.01 training.distribution=gaussian model.prior=0 training.risk=$r model.pred=$prd is_using_wandb=$wandb training.seed=$seed
done