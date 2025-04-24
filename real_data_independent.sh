for r in exact FO SO Rnd
do
	for d in HABER TTT SVMGUIDE CODRNA ADULT MUSH PHIS
	do
		for p in 1, 0.1, adjusted
		do
		  python3 real.py dataset=$d model.M=10 model.prior=p training.risk=$r model.pred=stumps-uniform
	  done
	done
done
