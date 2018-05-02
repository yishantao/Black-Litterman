# -*- coding: utf-8 -*-

"""
This module is used to implement Black-Litterman model.
"""

import numpy as np

from scipy import linalg


def blacklitterman(returns, tau, P, Q):
    """计算加入主观观点后的后验分布之期望值、协方差"""
    mu = returns.mean()  #
    sigma = returns.cov()
    pi1 = mu
    ts = tau * sigma
    Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
    middle = linalg.inv(np.dot(np.dot(P, ts), P.T) + Omega)
    er = np.expand_dims(pi1, axis=0).T + np.dot(np.dot(np.dot(ts, P.T), middle),
                                                (Q - np.expand_dims(np.dot(P, pi1.T), axis=1)))
    posteriorSigma = sigma + ts - np.dot(ts.dot(P.T).dot(middle).dot(P), ts)
    return [er, posteriorSigma]
