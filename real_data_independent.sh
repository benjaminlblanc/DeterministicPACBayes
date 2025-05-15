for distr in categorical dirichlet gaussian
do
  for r in FO SO Bin
  do
    for d in MUSH SVMGUIDE HABER TTT CODRNA ADULT PHIS
    do
      for p in adjusted 0 1
      do
        for s_i in ones rand
        do
          python3 real.py dataset=$d model.M=10 training.distribution=$distr model.prior=$p training.risk=$r model.pred=stumps-uniform is_using_wandb=True model.stump_init=$s_i
        done
      done
	  done
	done
done
