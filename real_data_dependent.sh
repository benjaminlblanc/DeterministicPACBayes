for distr in categorical dirichlet gaussian
do
  for r in FO SO Bin
  do
    for d in SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION
    do
      for p in adjusted 0 1
      do
        for bin_n in 0 10 20 100
        do
          python3 real.py dataset=$d model.M=100 training.distribution=$distr model.prior=$p training.risk=$r model.pred=rf is_using_wandb=True training.rand_n=$bin_n
        done
      done
	  done
	done
done
