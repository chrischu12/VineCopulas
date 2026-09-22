# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:39:03 2024
"""

# MODIFIED FILE. Original: VineCopulas 2.0.2 by Judith Claassen (VU-IVM),
# https://github.com/VU-IVM/VineCopulas, GPL-3.0.
# Modified 2025-11-19 to 2026-08-26 by Christian Jobelius Schulz for the replication
# package of "Online Conditional Vine Copulas: Forecasting Electricity Demand":
# reduced to the copula families and h-functions used in the paper, with
# covariate-dependent pair-copula parameters.
# SPDX-License-Identifier: GPL-3.0-only

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import rankdata
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.linalg import cholesky
from itertools import product
import sys
import os
import importlib
from scipy import optimize
import re


from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.links import FisherZLink, GaussianParameterToKendallsTau, Log, ClaytonParameterToKendallsTau, LogShiftTwo, GumbelLink, GumbelParameterToKendallsTau
from ondil.distributions import BivariateCopulaNormal, Normal, BivariateCopulaClayton, BivariateCopulaStudentT, BivariateCopulaGumbel

#%% Copulas

copulas = {
    0: "Independence",
    1: "Gaussian",
    2: "StudentT",
    31: "Clayton I",
    32: "Clayton II",
    33: "Clayton III",
    34: "Clayton IV",
    41: "Gumbel I",
    42: "Gumbel II",
    43: "Gumbel III",
    44: "Gumbel IV",
}

copula_distributions_bivariate = {

    0: "Independence",

    1: BivariateCopulaNormal(
    link=FisherZLink(),
    param_link=GaussianParameterToKendallsTau()
), 
    2: BivariateCopulaStudentT(
    link_1 = FisherZLink(),
    link_2 = LogShiftTwo(),
    param_link_1 = GaussianParameterToKendallsTau(),
    param_link_2 = GaussianParameterToKendallsTau(),
),  
    31: BivariateCopulaClayton(
    link= Log(),
    param_link=ClaytonParameterToKendallsTau(),
    family_code = 31,
), 
    32: BivariateCopulaClayton(
    link= Log(),
    param_link=ClaytonParameterToKendallsTau(),
    family_code = 32,
), 
    33: BivariateCopulaClayton(
    link= Log(),
    param_link=ClaytonParameterToKendallsTau(),
    family_code = 33,
), 
    34: BivariateCopulaClayton(
    link= Log(),
    param_link=ClaytonParameterToKendallsTau(),
    family_code = 34,
), 

    41: BivariateCopulaGumbel(
    link=GumbelLink(),
    param_link=GumbelParameterToKendallsTau(),
    family_code=41,
),

    42: BivariateCopulaGumbel(
    link=GumbelLink(),
    param_link=GumbelParameterToKendallsTau(),
    family_code=42,
),
    43: BivariateCopulaGumbel(
    link=GumbelLink(),
    param_link=GumbelParameterToKendallsTau(),
    family_code=43,
),    

    44: BivariateCopulaGumbel(
    link=GumbelLink(),
    param_link=GumbelParameterToKendallsTau(),
    family_code=44,
),
}

# Dynamic selection of copulas by their numeric codes

def select_copulas(codes):
    _all_copula_distributions = copula_distributions_bivariate.copy()
    selected_codes = set(codes)
    selected_codes.add(0)
    return {c: _all_copula_distributions[c] for c in selected_codes if c in _all_copula_distributions}

# Reverse mapping: distribution instance type to copula number
def get_copula_number(distribution_instance):
    """
    Returns the copula number for a given distribution instance.
    
    Arguments:
        distribution_instance: An instance of a copula distribution class
        
    Returns:
        The copula number (int) or None if not found
    """
    type_name = type(distribution_instance).__name__
    
    for cop_num, dist in copula_distributions_bivariate.items():
        if type(dist).__name__ == type_name:
            # For distributions with family_code, match both type and family_code
            if hasattr(dist, 'family_code') and hasattr(distribution_instance, 'family_code'):
                if dist.family_code == distribution_instance.family_code:
                    return cop_num
            else:
                # For distributions without family_code (e.g., Gaussian, StudentT)
                return cop_num
    
    return None


#%% best fit


def bestcop(cops, u, X, X_cols, early_stopped=False, edge=None, forget = None, method = None, fit_intercept = False, equation = None, scale_inputs = False):
    """
    Fits the best copula to data based on a selected list of copulas to fit to using the AIC.
    
    Arguments:
        *cops* : A list of integers referring to the copulae of interest for which the fit has to be evaluated. eg. a list of [1, 10] refers to the Gaussian and Frank copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit and evaluated. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
    Returns:  
     *cop* : An integer referring to the copula with the best fit. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *par* : The correlation parameters of the copula with the best fit, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *aic* : The Akaike information criterion of the copula with the best fit.
      
    """
    # Example usage:
    copula_distributions_bivariate = select_copulas(cops)

    if early_stopped==True:
        AIC = []
        PAR = []
        COEF = []
        LOGLIK = []
        ESTIM = []

        n_rows = u.shape[0]
        n_cols = X.shape[1]
        #for cop in cops:
            #par = fit(cop, u)
            #if cop == 15:
                #AIC.append(4 + (2 * neg_likelihood(par,cop,u)))
                #PAR.append(par)
            #else:
                #AIC.append(2 + (2 * neg_likelihood(par,cop,u)))
                #PAR.append(par)

        LOGLIK.append(0)
        AIC.append(0)
        PAR.append(np.zeros(n_rows))
        COEF.append(np.zeros(n_cols))
        ESTIM.append(None)

        i = np.where(AIC == np.nanmin(AIC))[0][0]  
        cop = 0
        distribution = "Independence"
        par = PAR
        aic = AIC
        coef = COEF
        loglik = LOGLIK
        estim = ESTIM


    else:   
        AIC = []
        PAR = []
        COEF = []
        LOGLIK = []
        ESTIM = []

        # Resolve the equation for this edge. It may be supplied directly, as a
        # callable ``(X_cols, edge) -> equation`` (e.g. for edge-specific covariate
        # selection), or left as None to use all available covariates.
        if callable(equation):
            equation = equation(X_cols, edge)
        elif equation is None:
            equation = {0: {0: np.arange(X.shape[1])}}

        for cop in cops:
            estimator = MultivariateOnlineDistributionalRegressionPath(
                distribution=copula_distributions_bivariate[cop],
                equation=equation,
                method=method,
                early_stopping=False,
                early_stopping_criteria="bic",
                iteration_along_diagonal=False,
                verbose=3,
                max_iterations_inner=20,
                max_iterations_outer=1,
                scale_inputs=scale_inputs,
                fit_intercept=fit_intercept,
                forget=forget,
                approx_fast_model_selection=False,
            )
            estimator.fit(X, u)
            if cop == 2:
                par = estimator.predict_distribution_parameters(X)
            else:
                par = estimator.predict(X)
            AIC.append(2 + (2 * -estimator._current_likelihood))
            PAR.append(par)
            LOGLIK.append(estimator._current_likelihood)
            COEF.append(estimator.coef_)
            ESTIM.append(estimator)





        i = np.where(AIC == np.nanmin(AIC))[0][0]  
        cop = cops[i]
        distribution = copula_distributions_bivariate[cop]
        par = PAR[i]
        aic = AIC[i]
        coef = COEF[i]
        loglik = LOGLIK[i]
        estim = ESTIM[i]

    return cop, distribution, par, aic, coef, loglik, estim



def bestcop_online(estimator, u, X):
    """
    Fits the best copula to data based on a selected list of copulas to fit to using the AIC.
    
    Arguments:
        *cops* : A list of integers referring to the copulae of interest for which the fit has to be evaluated. eg. a list of [1, 10] refers to the Gaussian and Frank copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit and evaluated. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
    Returns:  
     *cop* : An integer referring to the copula with the best fit. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *par* : The correlation parameters of the copula with the best fit, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
         
     *aic* : The Akaike information criterion of the copula with the best fit.
      
    """
    
    AIC = []
    PAR = []
    COEF = []
    LOGLIK = []
    ESTIM = []
    if estimator == None:
        cop = 0
        distribution = "Independence"
        par = np.zeros(u.shape[0])
        aic = 0
        coef = np.zeros(X.shape[1])
        loglik = 0
        estim = None   # <-- add this
    else: 
        cop = get_copula_number(estimator.distribution)

        estimator.update(X, u)

        if cop == 2:
            par = estimator.predict_distribution_parameters(X) 
        else:
            par = estimator.predict(X)

        
        AIC.append(2 + (2 * -estimator._current_likelihood))
        PAR.append(par)
        LOGLIK.append(estimator._current_likelihood)
        COEF.append(estimator.coef_)
        ESTIM.append(estimator)
        cop = get_copula_number(estimator.distribution)
        distribution = estimator.distribution
        par = PAR
        aic = AIC
        coef = COEF
        loglik = LOGLIK
        estim = ESTIM[0]


    return cop, distribution, par, aic, coef, loglik, estim


        
#%% h function

def hfunc(cop, u1, u2, par, un = 1, distribution = None):
    """
    Computes the h-function (conditional CDF) of a copula with respect to variable u1 or u2.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *u1* : A 1-d numpy array containing the samples of variable u1
        
        *u2* : A 1-d numpy array containing the samples of variable u2
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *un* : indicated with respect to which variable the h-function has to be calculated. if un = 1, the h-function is calculated with respect to u1 (c(u2|u1)), if un = 2, the h-function is calculated with respect to u2 (c(u1|u2)).
     
    Returns:  
     *y* : A 1-d numpy array containing the h-function of the copula evaluated with respect to u1 or u2.
      
    """
    try:
        
        u1[u1 <=0] = 0.000001
        u2[u2 <=0] = 0.000001
        u2[u2 >=1] = 0.999999
        u1[u1 >=1] = 0.999999   
    except:
        if u1 < 0.0001:
            u1 = 0.0001
        if u2 > 0.9999:
            u2 = 0.9999
        if u2 < 0.0001:
            u2 = 0.0001
        if u1 > 0.9999:
            u1 = 0.9999
    

    if cop == 0: 
        if un == 1:
            y = u2
            
        if un == 2:
            y = u1
    else:
        distribution = copula_distributions_bivariate[cop]
        rho = par

        if un == 2:
            y = distribution.hfunc(u1,u2,rho, un, family_code = cop)
            
        if un == 1:
            y = distribution.hfunc(u2,u1,rho, un, family_code = cop)

  

    try:
        if y < 0.0001:
            y = 0.0001
        if y > 0.9999:
            y = 0.9999
    except:
        y[y < 0.0001] = 0.0001
        y[y > 0.9999] = 0.9999
   
    return y 

#%% h-inverse func

def hfuncinverse(cop, ui, y, par, un = 1, distribution = None):
    """
    Computes the inverse h-function (inverse conditional CDF) of a copula with respect to variable u1 or u2.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2.
        
        *y* : A 1-d numpy array containing the h-function of the copula evaluated with respect to u1 or u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *un* : indicated with respect to which variable the h-function has to be calculated. if un = 1, the h-function is calculated with respect to u1 (c(u2|u1)), if un = 2, the h-function is calculated with respect to u2 (c(u1|u2)).
     
    Returns:  
     *uii* : A 1-d numpy array containing the inverse h-function of the copula evaluated with respect to u1 or u2.
      
    """
    
    if cop == 0:
        rho = par
        x1 = y
        x2 = ui
        #inner =( x1 * np.sqrt(1-rho**2)) + (rho * x2)
        #uii = st.norm.cdf(inner)

        rho = par
        if un == 1:
            uii = x2
            
        if un == 2:
            uii = x2
    else:
        distribution = copula_distributions_bivariate[cop]
        rho = par.copy()
        x1 = ui
        x2 = y
        if un == 1:
            uii = distribution.hinv(x1,x2,rho, un, family_code =cop)
            
        if un == 2:
            uii = distribution.hinv(x2,x1,rho, un, family_code =cop)

    
    
    try:
        if uii < 0.0001: 
            uii = 0.0001
        if uii > 0.9999:
            uii = 0.9999 
    except:
        uii[uii < 0.0001] = 0.0001
        uii[uii > 0.9999] = 0.9999
   
       
    return uii

#%% negative likelihood
def neg_likelihood(par,cop,u):
    """
    Computes the negative likelihood function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *l* : The negative likelihood as a scalar value.
      
    """
    l = -np.sum(np.log(PDF(cop, u, par)))
    return l
