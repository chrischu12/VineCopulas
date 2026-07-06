# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:39:03 2024


"""

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
from ondil.links import FisherZLink, ParameterToKendallsTau, Log, ClaytonParameterToKendallsTau, LogShiftTwo, GumbelLink, GumbelParameterToKendallsTau
from ondil.distributions import BivariateCopulaNormal, Normal, BivariateCopulaClayton, BivariateCopulaStudentT, BivariateCopulaGumbel
from online_copula_experiments.common import (
 TO_SCALE_COP
)

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
    param_link=ParameterToKendallsTau()
), 
    2: BivariateCopulaStudentT(
    link_1 = FisherZLink(),
    link_2 = LogShiftTwo(),
    param_link_1 = ParameterToKendallsTau(),
    param_link_2 = ParameterToKendallsTau(),
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




#%% fitting

def fit_vine_copulas(cop, u):
    """
    Fits a specific copula to data.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the copulae will be fit. Column 1 contains variable u1, and column 2 contains variable u2.
     
     
    Returns:  
     *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
      
    """
    u[u==1] = 0.999999
    u[u==0] = 0.000001


    #Gaussian
    if cop == 1:
        par = np.corrcoef(st.norm.ppf(u),rowvar=False)[0][1]
        
    # Gumbel all rotations
    if cop > 1 and cop < 6:
        res = minimize_scalar(neg_likelihood, bounds=(1, 20), args=(cop, u), method='bounded')
        par = res.x
    # Clayton all rotations
    if cop > 5 and cop < 10:
        res = minimize_scalar(neg_likelihood, bounds=(-1, 20), args=(cop, u), method='bounded')
        par = res.x
    # Frank
    if cop == 10:
       res = minimize_scalar(neg_likelihood, bounds=(-20, 20), args=(cop, u), method='bounded')
       par = res.x 
       
    # Joe all rotiations
    if cop > 10 and cop < 15:
        res = minimize_scalar(neg_likelihood, bounds=(1, 20), args=(cop, u), method='bounded')
        par = res.x
     
    # Student    
    if cop == 15:
        u1 = u[:,0]
        u2 = u[:,1]
        rho = np.sin(0.5 * np.pi * st.kendalltau(u1,u2)[0])
        R = np.array([[1, rho],
                      [rho, 1]])

        # Perform Cholesky decomposition
        R= np.linalg.cholesky(R)

        def invcdf(p):
            if p <= 0.9:
                q = (1000 / 9) * p
            else:
                q = 100 * (1 - 10 * (p - 0.9))**-5
            return q

        def negloglike(mu_, u, R):
            nu_ = invcdf(mu_)
            t_values = st.t.ppf(u, nu_)
            tRinv =np.linalg.solve(R, t_values.T).T
            
            n,d = u.shape
            
            nll = -n * gammaln((nu_ + d) / 2) + n * d * gammaln((nu_ + 1) / 2) - n * (d - 1) * gammaln(nu_ / 2) \
                  + n * np.sum(np.log(np.abs(np.diag(R)))) \
                  + ((nu_ + d) / 2) * np.sum(np.log(1 + np.sum(tRinv ** 2, axis=1) / nu_)) \
                  - ((nu_ + 1) / 2) * np.sum(np.sum(np.log(1 + t_values ** 2 / nu_), axis=1), axis=0)
            
            return nll


        res =minimize_scalar(negloglike, args=(u,R), 
             bounds=(0,1), method='bounded')
        df = invcdf(res.x)
        par = [rho, df]
    return par

#%% best fit


def bestcop(cops, u, X, X_cols, early_stopped=False, edge=None, application = None, t=None, forget = None, method = None, fit_intercept = False, equation = None):
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

        def get_index(cols, edge) -> np.ndarray:
            """
            Select columns for a given edge:
            - edge-specific columns (r1_r2_ or r2_r1_)
            - PLUS all non-pair-specific columns

            cols: DataFrame or array-like of column names
            edge: tuple/list of ints, e.g. (0, 1)
            """
            if isinstance(cols, pd.DataFrame):
                cols_str = cols.columns.astype(str).to_numpy()
            else:
                cols_str = np.asarray(cols, dtype=str)

            regions = ["1", "2", "3", "4", "5"]
            r1 = regions[int(edge[0])]
            r2 = regions[int(edge[1])]

            # edge-specific prefixes
            prefix_12 = f"{r1}_{r2}_"
            prefix_21 = f"{r2}_{r1}_"

            is_edge = (
                np.char.startswith(cols_str, prefix_12)
                | np.char.startswith(cols_str, prefix_21)
            )

            # detect any pair-specific prefix like "1_2_", "3_5_", etc.
            is_any_pair = np.array(
                [bool(re.match(r"^\d+_\d+_", c)) for c in cols_str]
            )

            # keep edge-specific OR non-pair-specific
            return is_edge | (~is_any_pair)
        
        if application == True:

            if t == 0:
                if equation == None:
                    equation = {0: { 0:np.arange(X.shape[1])[get_index(X_cols, edge = edge)]}}
                else: 
                    equation = equation

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
                        scale_inputs=TO_SCALE_COP,
                        fit_intercept=fit_intercept,
                        forget = forget,
                        approx_fast_model_selection = False ,
                        )
                    
                   # try:
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
                    #except Exception:
                      #  AIC.append(np.array([np.inf]))
                       # PAR.append(np.zeros(u.shape[0]).reshape(-1, 1))
                       # LOGLIK.append(np.array([-np.inf]))
                       # COEF.append(np.zeros(X.shape[1]))
                       # ESTIM.append(None)
            else:  
                equation = {  
                       0: { 0: "intercept"
                       }
               }
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
                        scale_inputs=TO_SCALE_COP,
                        fit_intercept=fit_intercept,
                        forget = forget,
                        approx_fast_model_selection = False ,
                        )
                    
                    #try:
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
                    #except Exception:
                      #  AIC.append(np.array([np.inf]))
                      #  PAR.append(np.zeros(u.shape[0]).reshape(-1, 1))
                      #  LOGLIK.append(np.array([-np.inf]))
                      #  COEF.append(np.zeros(X.shape[1]))
                      #  ESTIM.append(None)
      
        else:
            if t == 0:
                equation = {
                        0: {
                            h: np.arange(X.shape[1])
                            for h in range(1)
                            }
                }
                
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
                        #scale_inputs=np.array([False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]),
                        scale_inputs=True,
                        fit_intercept=fit_intercept,
                        forget = forget,
                        approx_fast_model_selection = False ,
                        )
                    #try:
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
                    #except Exception:
                        #if len(AIC) > 0:
                        #    AIC.append(AIC[-1])
                        #    PAR.append(PAR[-1])
                        #    LOGLIK.append(LOGLIK[-1])
                        #    COEF.append(COEF[-1])
                        #    ESTIM.append(ESTIM[-1])
                        #else:
                        #    n_rows = u.shape[0]
                        #    n_cols = X.shape[1]
                        #    AIC.append(0)
                        #    PAR.append(np.zeros(n_rows).reshape(-1,1))
                        #    LOGLIK.append(0)
                        #    COEF.append(np.zeros(n_cols))
                        #    ESTIM.append(None)
            else:  
                equation = {  
                        0: { h: "all"
                            for h in range(1)
                        }
                }
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
                        #scale_inputs=np.array([False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]),
                        scale_inputs=True,
                        fit_intercept=fit_intercept,
                        forget = forget,
                        approx_fast_model_selection = False ,
                        )
                    #try:
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
                    #except Exception:
                        #if len(AIC) > 0:
                        #    AIC.append(AIC[-1])
                        #    PAR.append(PAR[-1])
                        #    LOGLIK.append(LOGLIK[-1])
                        #    COEF.append(COEF[-1])
                        #    ESTIM.append(ESTIM[-1])
                        #else:
                        #    n_rows = u.shape[0]
                        #    n_cols = X.shape[1]
                        #    AIC.append(0)
                        #    PAR.append(np.zeros(n_rows).reshape(-1,1))
                        #    LOGLIK.append(0)
                        #    COEF.append(np.zeros(n_cols))
                        #    ESTIM.append(None)




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





#%%Copula random

def random(cop, par, n): 
    """
    Generates random numbers from a chosen copula with specific parameters.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
        *n* : Number of random samples to return, specified a positive integer.
     
    Returns:    
     *u* : A 2-d numpy array containing random samples with n amount of rows. Column 1 contains variable u1, and column 2 contains variable u2.
      
    """

    
    # Gaussian
    if cop == 1:
        rho = par
        rho = np.array([[1, rho], [rho,1]])
        u = st.norm.cdf(np.random.multivariate_normal(np.array([0, 0]), rho, n))
    
    if cop > 1:
        alpha = par
        
    # Gumbel 
    if cop > 1 and cop < 6:
        v1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        def equation2(w):
            return (w * (1-(np.log(w)/alpha))) - v2
        w_guess = v2
        w = newton(equation2, w_guess,maxiter=2000)
        u1 = np.exp((v1**(1/alpha)) * np.log(w))
        u2 = np.exp(((1-v1)**(1/alpha)) * np.log(w))
       
        # 90 dergees
        if cop == 3:
            u1 = 1 - u1
        # 180 dergees
        elif cop == 4:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 dergees
        elif cop == 5:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
        
    # Clayton
    if  cop > 5 and cop < 10:
        u1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        u2 = ((u1**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 90 degrees
        if cop == 7:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 8:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 9:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
    
    # Frank
    if cop == 10:
        ui = np.random.rand(n)
        y = np.random.rand(n)
        uii = (-1/alpha)*np.log(1+((y*(1-np.exp(-alpha)))/(y*(np.exp(-alpha*ui)-1)-np.exp(-alpha*ui))))
        u = np.vstack((ui,uii)).T
    
    # Joe
    if  cop > 10 and cop < 15:
        v1 = np.random.uniform(0.00001,0.999999,n)
        v2 = np.random.uniform(0.00001,0.999999,n)
        def equation2(w):
            return w - ((1/alpha) * ((np.log((1-(1-w)**alpha))*(1-(1-w)**alpha))/((1-w)**(alpha-1)))) - v2
        w_guess = v2
        w = newton(equation2, w_guess,maxiter=2000)
        u1 = 1 - (1-(1-(1-w)**alpha)**v1)**(1/alpha)
        u2 =  1 - (1-(1-(1-w)**alpha)**(1-v1))**(1/alpha)
        # 90 degrees
        if cop == 12:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 13:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 14:
            u2 = 1 - u2
        u = np.vstack((u1,u2)).T
    
    
    #Student
    if cop == 15:
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
               [alpha, 1.       ]])
        df = par[1]
        k = st.multivariate_t(np.array([0, 0]), shape = alpha, df = df)
        u = st.t.cdf(k.rvs(size=n), df = df)
    
    u[u <=0] = 0.000001
    u[u >=1] = 0.999999
    
  
    
    return u
        
        
#%% Copula conditional random
       
def randomconditional(cop, ui, par, n, un = 1):

    """
    Generates conditional random numbers from a chosen copula with specific parameters.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *ui* : A 1-d numpy array containing the samples of variable u1, if evaluated with respect to u1, or u2 if evaluated with respect to u2 on which conditional samples should be computed
        
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__)
        
        *n*: number of samples to draw 
        
        *un* : indicated with respect to which variable the conditional samples have to be drawn. if un = 1, conditional samples of u2 will be drawn based on u1, if un = 2, conditional samples of u1 will be drawn based on u2.
     
    Returns:  
        *uii* : A 1-d numpy array containing the inverse h-function of the copula evaluated with respect to u1 or u2.
      
    """

    # Gaussian
    if cop == 1:
        ui = np.full(shape = n, fill_value = ui)
        y  = np.random.uniform(0,1,n)
        x1 = st.norm.ppf(y)
        x2 = st.norm.ppf(ui)
        inner =( x1 * np.sqrt(1-par**2)) + (par * x2)
        uii = st.norm.cdf(inner)
    
    if cop > 1:
        alpha = par
        
    # Gumbel 
    if cop > 1 and cop < 6:
        # 90 degrees
        if cop == 3 and un == 1:
            ui = 1- ui
        # 180 degrees
        elif cop == 4:
            ui = 1 - ui
        # 270 degrees
        elif  cop == 5 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        vi = np.random.uniform(0.00001,0.999999,n)
        w = np.exp(np.log(ui)/ (vi **(1/alpha)))
        uii = np.exp(((1-vi)**(1/alpha)) * np.log(w))
        # 90 degrees
        if cop == 3 and un == 2:
            uii = 1 - uii
        # 180 degrees
        elif cop == 4: 
            uii = 1 - uii
        # 270 degrees
        elif cop == 5  and un == 1:
            uii = 1 - uii
        
        
    # Clayton
    if  cop > 5 and cop < 10:
        # 90 degrees
        if cop == 7 and un == 1:
            ui = 1- ui
        # 180 degrees
        elif cop== 8:
            ui = 1 - ui
        # 270 degrees
        elif cop == 9 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        v2 = np.random.uniform(0.00001,0.999999,n)
        uii = ((ui**-alpha)*((v2**(-alpha/(1+alpha)))-1)+1)**(-1/alpha)
        # 90 degrees
        if cop == 7 and un == 2:
            uii = 1 - uii
        # 180 degrees
        elif cop == 8: 
            uii = 1 - uii
        # 270 degrees
        elif cop == 9  and un == 1:
            uii = 1 - uii
        
        try:
            if np.isnan(uii) == True:
                uii =  0.00001
            if uii < 0.00001:
                uii =  0.00001
        except:
            uii[np.isnan(uii)] = 0.00001
            uii[uii<0.00001] = 0.00001
        
  
 
    
    # Frank
    if cop == 10:
        ui = np.full(shape = n, fill_value = ui)
        p = np.random.rand(n)

        if abs(alpha) > np.log(sys.float_info.max):
            uii = (ui < 0) + np.sign(alpha) * ui  # u1 or 1-u1
        elif abs(alpha) > np.sqrt(np.finfo(float).eps):
            uii = -np.log((np.exp(-alpha * ui) * (1 - p) / p + np.exp(-alpha)) / (1 + np.exp(-alpha * ui) * (1 - p) / p)) / alpha
        else:
            uii = p
        

    # Joe
    if  cop > 10 and cop < 15:
        # 90
        if cop == 12 and un == 1:
            ui = 1- ui
        # 180
        elif cop == 13:
            ui = 1 - ui
        # 270
        elif cop == 14 and un == 2:
            ui = 1 - ui
        ui = np.full(shape = n, fill_value = ui)
        v1 = np.random.uniform(0.00001,0.999999,n)
        if un == 1:
            w = 1 - (1-(1-(1-ui)**(alpha))**(1/v1))**(1/alpha)
            uii =  1 - (1-(1-(1-w)**alpha)**(1-v1))**(1/alpha)
        elif un == 2:
            w =  1 - (1-(1-(1-ui)**alpha)**(1/(1-v1)))**(1/alpha)
            uii = 1 - (1-(1-(1-w)**alpha)**v1)**(1/alpha)
        # 90
        if cop == 12 and un == 2:
            uii = 1 - uii
        # 180
        elif cop == 13: 
            uii = 1 - uii
        # 270
        elif cop == 14  and un == 1:
            uii = 1 - uii
    
    
    #Student
    if cop == 15:
        ui = np.full(shape = n, fill_value = ui)
        y  = np.random.uniform(0,1,n)
        alpha = par[0]
        df = par[1]
        xi = st.t.ppf(ui, df) 
        xy= st.t.ppf(y, df) 
        inner = xy * np.sqrt(((df+xi**2)*(1-alpha**2))/(df+1)) + (alpha*xi)
        uii = st.t.cdf(inner, df = df)
    
    uii[uii <=0] = 0.000001
    uii[uii >=1] = 0.999999
    
  
    
    return uii

#%% CDF
def CDF(cop, u, par):
    
    """
    Computes the cumulative distribution function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the CDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *p* : A 1-d numpy array containing the cumulative distribution function of the copula evaluated at u1 and u2.
      
    """
    # Gaussian
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 1: 
        rho = par
        rho = np.array([[1.       , rho],
               [rho, 1.       ]])
        _, d = u.shape

        p = st.multivariate_normal.cdf(st.norm.ppf(u), mean=np.zeros(d), cov=rho)
 
    if cop > 1:
        u1 = u[:,0]
        u2 = u[:,1]
        alpha = par
    # Gumbel 0 degrees
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 2:
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 90 degrees
    elif cop == 3:
        u1 = 1 - u1
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u2 - np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 180 degrees
    elif cop == 4:
        u1_u2 = u1 + u2
        u1 = 1 - u1
        u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u1_u2 -1 + np.exp(-(t1+t2)**(1/alpha))
    # Gumbel 270 degrees
    elif cop == 5:
        u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        p = u1- np.exp(-(t1+t2)**(1/alpha))
        
    # Clayton 0 degrees  
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 6:
        p = ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
        try:
            if np.isnan(p) == True:
                p =  0.00001
            if p < 0.00001:
                p =  0.00001
        except:
            p[np.isnan(p)] = 0.00001
            p[p<0.00001] = 0.00001
            
    # Clayton 90 degrees
    elif cop == 7:
        u1 = 1 - u1
        p = u2 - ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
        try:
            if np.isnan(p) == True:
                p =  0.00001
            if p < 0.00001:
                p =  0.00001
        except:
            p[np.isnan(p)] = 0.00001
            p[p<0.00001] = 0.00001
    # Clayton 180 degrees
    elif cop == 8:
        u1_u2 = u1 + u2
        u2 = 1 - u2
        u1 = 1 - u1
        p =  u1_u2 -  1 + (((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha))
        try:
            if np.isnan(p) == True:
                p =  0.00001
            if p < 0.00001:
                p =  0.00001
        except:
            p[np.isnan(p)] = 0.00001
            p[p<0.00001] = 0.00001
    # Clayton 270 degrees
    elif cop == 9:
        u2 = 1 - u2
        p = u1 - ((u1 ** -alpha) + (u2 ** -alpha) - 1)**(-1/alpha)
        try:
            if np.isnan(p) == True:
                p =  0.00001
            if p < 0.00001:
                p =  0.00001
        except:
            p[np.isnan(p)] = 0.00001
            p[p<0.00001] = 0.00001
    
    # Frank
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 10:
        p = (-1/alpha)*np.log((1/(1-np.exp(-alpha)))*(1-np.exp(-alpha) - (1- np.exp(-alpha*u1))*(1- np.exp(-alpha*u2))))
    
    # Joe 0 degrees
    # Reference: Schepsmeier and Stöber, 2013
    elif cop == 11:
        p = 1- ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha)
    # Joe 90 degrees
    elif cop == 12:
        u1 = 1 - u1
        p = u2 - (1- ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha))
    # Joe 180 degrees
    elif cop == 13:
        u1_u2 = u1 + u2
        u2 = 1 - u2
        u1 = 1 - u1
        p =  u1_u2 -  1  +  1 - ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha)
    # Joe 270 degrees
    elif cop == 14:
        u2 = 1 - u2
        p = u1 -  (1 - ((1-u1)**alpha + (1-u2)**alpha - (1-u1)**alpha *(1-u2)**alpha ) ** (1/alpha))

    # Student
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 15:
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
            [alpha, 1.       ]])
        df = par[1]
        p = st.multivariate_t.cdf(st.t.ppf(u, df), shape = alpha, df = df)
    
    #BB1
   # if cop == 16:
      #  theta = par[0]
      #  delta = par[1]
      #  p = (1 + ((((u1**-theta) - 1) **delta) + (((u2**-theta) - 1) **delta))**(1/delta))**(-1/theta)
    return p

#%% PDF
    
def PDF(cop, u, par):
    
    """
    Computes the probability density function.
    
    Arguments:
        *cop* : An integer referring to the copula of choice. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
        
        *u* : A 2-d numpy array containing the samples for which the PDF will be calculated. Column 1 contains variable u1, and column 2 contains variable u2.
     
        *par* : The correlation parameters of the copula, provided as a scalar value for copulas with one parameter and as a list for copulas with more parameters (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).
     
    Returns:  
     *y* : A 1-d numpy array containing the probability density function of the copula evaluated at u1 and u2.
      
    """
    u[u <= 0 ] = 0.00001
    u[u >= 1 ] = 0.99999
    # Gaussian
    # Reference: Schepsmeier and Stöber, 2013
    if cop == 1: 
        rho = par
        x1 = st.norm.ppf(u[:,0])
        x2 = st.norm.ppf(u[:,1])
        rhosq = rho **2 
        y = (1/np.sqrt(1 -  rhosq )) * np.exp( -(rhosq* (x1**2 + x2**2) - (2* rho * x1 * x2))/( 2* (1-rhosq)))
 
    if cop > 1:
        u1 = u[:,0]
        u2 = u[:,1]
        alpha = par
    # Gumbel
    # Reference: Schepsmeier and Stöber, 2013
    if cop > 1 and cop < 6:    
        # 90 dergees
        if cop == 3:
            u1 = 1 - u1
        # 180 dergees
        elif cop == 4:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 dergees
        elif cop == 5:
            u2 = 1 - u2
        t1 = (-np.log(u1))**alpha
        t2 = (-np.log(u2))**alpha
        cdf =  np.exp(-(t1+t2)**(1/alpha))
        y = cdf * (1/(u1*u2)) * ((t1+t2) **(-2 + (2/alpha)))*((np.log(u1) * np.log(u2))**(alpha-1))* (1+(alpha - 1)*(t1+t2)**(-1/alpha))
    
    # Clayton
    # Reference: Schepsmeier and Stöber, 2013
    if  cop > 5 and cop < 10:
        # 90 degrees
        if cop == 7:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 8:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 9:
            u2 = 1 - u2
        y = ((1+alpha) * (u1 * u2) ** (-1 - alpha)) / ( ((u1 ** -alpha) + (u2 ** -alpha) - 1)**((1/alpha)  + 2))

    
    # Frank
    elif cop == 10:
        y = alpha*(-np.exp(alpha*(u1 + u2)) + np.exp(alpha*(u1 + u2 + 1)))*np.exp(alpha)/((1 - np.exp(alpha*u1))*(1 - np.exp(alpha*u2))*np.exp(alpha) + np.exp(alpha*(u1 + u2)) - np.exp(alpha*(u1 + u2 + 1)))**2

    if  cop > 10 and cop < 15:
        # 90 degrees
        if cop == 12:
            u1 = 1 - u1
        # 180 degrees
        elif cop == 13:
            u1 = 1 - u1
            u2 = 1 - u2
        # 270 degrees
        elif cop == 14:
            u2 = 1 - u2
        y = (1 - u1)**(alpha - 1)*(1 - u2)**(alpha - 1)*(-(1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha)**((1 - 2*alpha)/alpha)*(alpha - (1 - u1)**alpha*(1 - u2)**alpha + (1 - u1)**alpha + (1 - u2)**alpha - 1)         

    # Student
    if cop == 15:
        n, d = u.shape
        alpha = par[0]
        alpha = np.array([[1.       , alpha],
               [alpha, 1.       ]])
        df = par[1]
        R = cholesky(alpha, lower=True)
        t = st.t.ppf(u,df)
        z = np.linalg.solve(R, t.T).T
        logSqrtDetRho = np.sum(np.log(np.diag(R)))
        const = gammaln((df + d) / 2) + (d - 1) * gammaln(df / 2) - d * gammaln((df + 1) / 2) - logSqrtDetRho
        numer = -((df + d) / 2) * np.log(1 + np.sum(z**2, axis=1) / df)
        denom = np.sum(-((df + 1) / 2) * np.log(1 + (t**2) / df), axis=1)
        y = np.exp(const + numer - denom)
        
    return y
        
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

#%% negative likelyhood
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
