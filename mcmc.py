#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:52:56 2020

@author: mckay
"""

import numpy as np
from scipy.stats import norm, invgamma
from scipy.linalg import solve_triangular


def multnorm_sampler(mu, inverse_covar=None, L=None):
    '''A multinormal distribution sampler that uses the precision instead of the covariance
    Provide either the precision matrix or its cholesky decomp (lower triangular)
    '''
    z = norm.rvs(size=len(mu))
    if L is None:
        L = np.linalg.cholesky(inverse_covar)
    # solve L.T (beta - mu) = z for beta
    sol = solve_triangular(L.T, z)
    return sol + mu
    


def gibbs_sampler(X, y, beta0, sigma_squared0, B0inv, alpha0, delta0, burn_in, sample_size):
    '''Gibbs sampler for Bayesian normal linear regression model
    
    See Greenberg p. 116, adapted for better numerical stability
    '''
    n = len(X)
    k = len(beta0)
    assert X.shape == (n, k)
    alpha1 = alpha0 + n
    # set up beta and sigma_squared chains
    beta = np.zeros((burn_in + sample_size, k))
    beta[0] = beta0
    sigma_squared = np.zeros(burn_in + sample_size)
    sigma_squared[0] = sigma_squared0
    # compute some operations to save time later
    XT = X.T
    XTX = XT.dot(X)
    XTy = XT.dot(y)
    for i in range(1, burn_in + sample_size):
        if i % 1000 == 0 or i == burn_in + sample_size - 1:
            print(f'Sampling {i} / {burn_in + sample_size}')
        B1inv = XTX / sigma_squared[i-1] + B0inv
        # using cholesky decomp (since we need cholesky anyway), compute beta_bar
        L = np.linalg.cholesky(B1inv)
        beta_bar = solve_triangular(L.T, solve_triangular(L, XTy / sigma_squared[i-1] + B0inv.dot(beta0), lower=True))
        beta[i] = multnorm_sampler(beta_bar, L=L)
        Z = y - X.dot(beta[i])
        delta = delta0 + Z.T.dot(Z)
        sigma_squared[i] = invgamma.rvs(a=alpha1/2, scale=delta/2)
    return beta, sigma_squared
