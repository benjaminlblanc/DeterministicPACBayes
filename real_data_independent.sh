for r in exact
do
	for d in HABER TTT SVMGUIDE CODRNA ADULT MUSH PHIS
	do
		for p in adjusted
		do
		  python3 real.py dataset=$d model.M=10 model.prior=$p training.risk=$r model.pred=stumps-uniform
	  done
	done
done
