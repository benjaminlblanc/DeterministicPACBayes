import logging
import warnings
import numpy as np
from time import time

from core.Cbound.core.metrics import Metrics
from core.Cbound.learner.c_bound_joint_learner import CBoundJointLearner
from core.Cbound.voter.stump import DecisionStumpMV


###############################################################################


def C_bound_optimization(cfg, x_train, y_train, x_test, y_test):
    logging.basicConfig(level=logging.INFO)
    logging.StreamHandler.terminator = ""
    warnings.filterwarnings("ignore")
    t_init = time()
    # ----------------------------------------------------------------------- #

    zero_one = Metrics("ZeroOne").fit

    def generate_MV_stump(x_train, y_train):
        majority_vote = DecisionStumpMV(
            x_train, y_train,
            nb_per_attribute=cfg.model.M,
            complemented=True, quasi_uniform=False)
        return x_train, y_train, majority_vote

    voter = "decision stumps"
    epoch_dict = {"decision stumps": 1000}

    # ----------------------------------------------------------------------- #
    if len(y_train.shape) == 1:
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))

    # We generate the majority vote (MV)
    x_train, y_train, majority_vote = generate_MV_stump(x_train, y_train)

    # We learn the posterior distribution associated to the MV
    learner = CBoundJointLearner(majority_vote, epoch=epoch_dict[voter], batch_size=y_train.shape[0])
    learner = learner.fit(x_train, y_train)

    # We compute the train/test majority vote risk and the PAC-Bayesian C-Bound
    y_p_train = learner.predict(x_train)
    y_p_test = learner.predict(x_test)
    c_bound = Metrics("CBoundJoint", majority_vote, delta=cfg.bound.delta).fit
    r_MV_S = zero_one(y_train, y_p_train)
    r_MV_T = zero_one(y_test, y_p_test)
    cb = c_bound(y_train, y_p_train)

    logging.info(("MV train risk: {:.4f}\n").format(r_MV_S))
    logging.info(("PAC-Bayesian C-Bound: {:.4f}\n").format(cb))
    logging.info(("MV test risk: {:.4f}\n").format(r_MV_T))

    return cb, r_MV_S, r_MV_T, time() - t_init