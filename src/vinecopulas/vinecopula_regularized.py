# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:40:09 2024


"""


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.stats as st
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import os
import importlib
import sys


from vinecopulas.bivariate import *

# %% Copulas

copulas = {
    1: "Gaussian",
    2: "Gumbel0",
    3: "Gumbel90",
    4: "Gumbel180",
    5: "Gumbel270",
    6: "Clayton0",
    7: "Clayton90",
    8: "Clayton180",
    9: "Clayton270",
    10: "Frank",
    11: "Joe0",
    12: "Joe90",
    13: "Joe180",
    14: "Joe270",
    15: "Student",
    0: "Independence",
}

# %% fitting vinecopula

def fit_vinecop(u1, X, copsi, vine="R", printing=True, distribution=None, estimator=None, min_ll_increase=1e-3):
    """
    Fit a regular vine copula to data with early stopping based on log-likelihood improvement.

    Arguments:
        *u1* :  the data, provided as a numpy array where each column contains a separate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)

        *copsi* : A list of integers referring to the copulae of interest for which the fit has to be evaluated in the vine copula. eg. a list of [1, 10] refers to the Gaussian and Frank copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *vine* : The type of vine copula that needs to be fit, either 'R', 'D', or 'C'

        *printing*: True if the fitted copula should be printed and False if not
        
        *min_ll_increase*: Minimum log-likelihood increase required to continue to next tree (default: 1e-3)

    Returns:
     *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

     *p* : Parameters of the bivariate copulae provided as a triangular matrix.

     *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """
    # Reference: Dißmann et al. 2013

    v1 = []  # list for variable 1
    v2 = []  # list for  variable 2
    tauabs = []  # list for the absolute kendal tau between v1 and v2
    dimen = u1.shape[1]  # number of variables (number of columns)
    for i in range(dimen - 1):
        for j in range(i + 1, dimen):
            v1.append(int(i))  # add variable to v1
            v2.append(int(j))  # add variable to v2
            tauabs.append(
                abs(st.kendalltau(u1[:, i], u1[:, j])[0])
            )  # calculate the absolute kendall tau between v1 and v2 and add it to the taubs list

    order1 = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": tauabs}
    )  # put the v1, v2, and tauabs list into a dataframe for the first tree
    #order1 = order1.sort_values(by="tauabs", ascending=False).reset_index(
    #    drop=True
    #)  # sort this dataframe from highest to lowest tauabs
    print(order1)

    # R vine
    if vine == "R":
        inde = []  # list to put used rows of order1 in
        for i in range(
            len(order1)
        ):  # loop through all the rows to include those pairs with the highest ktau first
            if i == 0:
                order2 = order1.head(
                    1
                )  # create a new dataframe where the row with the highest ktau is the first row
            else:
                if (
                    order1.v1[i] in list(order1.v2[:i])
                    or order1.v1[i] in list(order1.v1[:i])
                ) and (
                    order1.v2[i] in list(order1.v2[:i])
                    or order1.v2[i] in list(order1.v1[:i])
                ):  # see if both v1i and v2i are already in order2, first rows with unique variables need to be added to ensure all variables are included
                    continue
                else:
                    inde.append(i)  # add used rows to inde
                    order2 = pd.concat(
                        [order2, order1.loc[i].to_frame().T], ignore_index=True
                    )  # add row to dataframe order2
        if len(order2) < (
            dimen - 1
        ):  # the first tree will have the number of variables - 1 number of rows.
            for i in range(
                1, len(order1)
            ):  # loop through all the combinations again and add based on the highest ktauabs
                if i in inde:  # check if row has not been added before
                    continue
                lst = (
                    list(order2.v2[order2.v1 == order1.v2[i]])
                    + list(order2.v1[order2.v2 == order1.v2[i]])
                    + list(order2.v2[order2.v1 == order1.v1[i]])
                    + list(order2.v1[order2.v2 == order1.v1[i]])
                )
                l1 = list(order2.v2[order2.v1 == order1.v2[i]]) + list(
                    order2.v1[order2.v2 == order1.v2[i]]
                )
                lk = l1.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v2[i])
                        except:
                            pass

                        for s in l1:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l1 = l1 + ln
                        lk = lk + ln

                l2 = list(order2.v2[order2.v1 == order1.v1[i]]) + list(
                    order2.v1[order2.v2 == order1.v1[i]]
                )
                lk = l2.copy()
                while len(lk) > 0:
                    lk2 = lk.copy()
                    lk = []
                    for j in lk2:
                        ln = list(order2.v2[order2.v1 == j]) + list(
                            order2.v1[order2.v2 == j]
                        )
                        try:
                            ln.remove(order1.v1[i])
                        except:
                            pass

                        for s in l2:
                            try:
                                ln.remove(s)
                            except:
                                pass
                        l2 = l2 + ln
                        lk = lk + ln
                skip = False
                for val in l1:
                    if val in l2:
                        skip = True
                        break
                if skip == True:
                    continue
                # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1): # ensure that there are no groups of variables that all have a connection with one another
                #  continue
                if len(lst) == len(
                    set(lst)
                ):  # check that there are no 'triangles', e.g. three variables that are alll connected with eachother
                    order2 = pd.concat(
                        [order2, order1.loc[i].to_frame().T], ignore_index=True
                    )  # add row if conditions are met
                if (
                    len(order2) == dimen - 1
                ):  # end loop when desired number of rows (number of variables - 1) has been met
                    break

    # D vine
    elif vine == "D":
        inde = []  # lsit to put used variables in
        for j in range(dimen - 1):
            if j == 0:
                order2 = order1.head(
                    1
                )  # loop through all the rows to include those pairs with the highest ktau first
                vi = order2.v1[0]  # select the first variable included in this row
                vj = order2.v2[0]  # select the second variable included in this row
            else:
                for i in range(len(order1)):
                    if (
                        order1.v1[i] == vj
                        or order1.v2[i] == vi
                        or order1.v1[i] == vi
                        or order1.v2[i] == vj
                    ):  # find the rows which include at least one of the two variables from the previous edge
                        if (
                            order1.v1[i] in inde or order1.v2[i] in inde
                        ):  # check if variables have already been connected to other variables twice
                            continue
                        if (order1.v1[i] == vi and order1.v2[i] == vj) or (
                            order1.v1[i] == vj and order1.v2[i] == vi
                        ):  # skip rows that have already been used
                            continue
                        else:
                            order2 = pd.concat(
                                [order2, order1.loc[i].to_frame().T], ignore_index=True
                            )  # if conditions are met, add this to the dataframe of the first tree
                            if (
                                order1.v1[i] == vj or order1.v2[i] == vj
                            ):  # check which value has been used twice already
                                inde.append(vj)  # add this value to inde
                                if order1.v1[i] == vj:
                                    vj = order1.v2[
                                        i
                                    ]  # select the new edge to connect to
                                else:
                                    vj = order1.v1[
                                        i
                                    ]  # select the new edge to connect to
                            elif (
                                order1.v1[i] == vi or order1.v2[i] == vi
                            ):  # check which value has been used twice already
                                inde.append(vi)  # add this value to inde
                                if order1.v1[i] == vi:
                                    vi = order1.v2[
                                        i
                                    ]  # select the new edge to connect to
                                else:
                                    vi = order1.v1[
                                        i
                                    ]  # select the new edge to connect to
    elif vine == "C":
        taus = []  # list to put the sum of all tauabs in for a specific variable
        for i in range(dimen):
            taus.append(
                sum(order1[(order1.v1 == i) | (order1.v2 == i)].tauabs)
            )  # calculate the sum of all taus for a specific variable
        i = np.where(np.array(taus) == max(taus))[0][
            0
        ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
        order2 = order1[(order1.v1 == i) | (order1.v2 == i)].reset_index(
            drop=True
        )  # select the rows that include this variable



    order1 = order2  # make order1 == the first tree
    del order2
    rhos = []  # list for the rhos
    coefs = []
    node = []  # list for the nodes
    cops = []  # list for the copulas
    v1_1 = []  # list for the 1st nodes
    v2_1 = []  # list for the 2nd nodes
    aics = []  # list for the AIC's
    logliks = []  # list for the log-likelihoods
    for i in range(len(order1)):
        v1i = int(order1.v1[i])  # first node
        v2i = int(order1.v2[i])  # second node
        u3 = np.vstack(
            (u1[:, v1i], u1[:, v2i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        coefs.append(coef)
        cops.append(cop)  # add copula to cops
        logliks.append(loglik)  # add log-likelihood to logliks
        node.append([v1i, v2i])  # create final node
        v1_1.append(u1[:, v1i])  # add array of first node
        v2_1.append(u1[:, v2i])  # add array of second node
    v1_1 = np.array(v1_1).T  # v1
    v2_1 = np.array(v2_1).T  # v2

    # add information to dataframe of the first tree
    order1["rhos"] = rhos
    order1["coefs"] = coefs
    order1["node"] = node
    order1["tree"] = 0
    order1["cop"] = cops
    order1["AIC"] = aics
    order1["loglik"] = logliks
    # Calculate initial log-likelihood for tree 1
    prev_loglik = np.sum(logliks)  # Initial log-likelihood for tree 1
    early_stopped = False
    max_tree = dimen - 1

    # set up variables for the second tree
    v1 = []
    v2 = []
    ktau = []
    rhos = []
    coefs = []
    node = []
    cops = []
    v1_k = []
    v2_k = []
    aics = []
    logliks = []
    for i in range(
        len(order1) - 1
    ):  # loop through the first tree to identify all possible combination of nodes
        v1i = int(order1.v1[i])  # parent node 1 from nodei in tree
        v2i = int(order1.v2[i])  # parent node 2 from nodei in tree
        copi = int(order1.cop[i])  # copula of nodei
        pari = order1.rhos[i]  # parameters of nodei
        for j in (
            np.where(
                np.array([item == v1i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v1i for item in list(order1.v2[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v1[i + 1 :])])
                | np.array([item == v2i for item in list(order1.v2[i + 1 :])])
            )[0]
            + i
            + 1
        ):  # see if possible connection between nodei and nodej
            v1j = int(order1.v1[j])  # parent node 1 from nodej in tree
            v2j = int(order1.v2[j])  # parent node 2 from nodej in tree
            copj = int(order1.cop[j])  # copula of nodei
            parj = order1.rhos[j]  # parameters of nodei
            v1.append(order1.node[i])  # parent node for next tree
            v2.append(order1.node[j])  # parent node for next tree
            lst = order1.node[i] + order1.node[j]  # list of all variables in node
            s = max(
                set(lst), key=lst.count
            )  # variable that is common in both parent nodes
            # parent node values
            ui1 = v1_1[:, i]
            ui2 = v2_1[:, i]
            uj1 = v1_1[:, j]
            uj2 = v2_1[:, j]
            # defining which parent node the conditional CDF needs to be based on
            if v1i == s:
                uni = 1
                vi1 = v2i
            else:
                uni = 2
                vi1 = v1i
            if v1j == s:
                unj = 1
                vj1 = v2j
            else:
                unj = 2
                vj1 = v1j
            # calculate the conditional CDF
            v1igs = hfunc(copi, ui1, ui2, pari, un=uni, distribution=distribution)
            v2jgs = hfunc(copj, uj1, uj2, parj, un=unj, distribution=distribution)
            ktau.append(
                abs(st.kendalltau(v1igs, v2jgs)[0])
            )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
            # add the parent node values
            v1_k.append(v1igs)
            v2_k.append(v2jgs)
            # format the final node
            node.append([vi1, vj1, "g", s])

    k = 2  # second tree

    orderk = pd.DataFrame(
        {"v1": v1, "v2": v2, "tauabs": ktau, "node": node}
    )  # put the v1, v2, tauabs, and the node in list into a dataframe for tree k

    if vine == "R" or vine == "D":
        if (
            len(orderk) > dimen - k
        ):  # the tree will have the number of variables - k number of rows, if this is exceeded in orderk, a selection has to be made based on kendal tau
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            indexes = list(
                orderk.index
            )  # create a list the original order of the dataframe prior to sorting
            orderk = orderk.reset_index(drop=True)  # reset the index
            inde = (
                []
            )  #  list to put used rows of orderk in based on their oroginal position in the dataframe
            inde2 = []  # list to put used rows of orderk i
            for i in range(len(orderk)):
                if i == 0:
                    order = orderk.head(
                        1
                    )  # loop through all the rows to include those pairs with the highest ktau first
                    inde.append(
                        indexes[i]
                    )  # add used rows to inde (original row value)

                else:
                    if (
                        orderk.v1[i] in list(orderk.v2[:i])
                        or orderk.v1[i] in list(orderk.v1[:i])
                    ) and (
                        orderk.v2[i] in list(orderk.v2[:i])
                        or orderk.v2[i] in list(orderk.v1[:i])
                    ):
                        continue
                    else:
                        inde.append(
                            indexes[i]
                        )  # add used rows to inde (original row value)
                        inde2.append(i)  # add used rows to inde
                        order = pd.concat(
                            [order, orderk.loc[i].to_frame().T], ignore_index=True
                        )  # if conditions are met, add this to the dataframe of tree k

            if len(order) < (dimen - k):
                for i in range(1, len(orderk)):
                    if i in inde2:
                        continue
                    lst = (
                        list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v1[i])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v1[i])
                            ]
                        )
                        + list(
                            order.v2.astype(str)[
                                order.v1.astype(str) == str(orderk.v2[i])
                            ]
                        )
                        + list(
                            order.v1.astype(str)[
                                order.v2.astype(str) == str(orderk.v2[i])
                            ]
                        )
                    )
                    l1 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v1[i])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v1[i])]
                    )
                    lk = l1.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v1[i]))
                            except:
                                pass

                            for s in l1:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l1 = l1 + ln
                            lk = lk + ln

                    l2 = list(
                        order.v2.astype(str)[order.v1.astype(str) == str(orderk.v2[i])]
                    ) + list(
                        order.v1.astype(str)[order.v2.astype(str) == str(orderk.v2[i])]
                    )
                    lk = l2.copy()
                    while len(lk) > 0:
                        lk2 = lk.copy()
                        lk = []
                        for j in lk2:
                            ln = list(
                                order.v2.astype(str)[order.v1.astype(str) == j]
                            ) + list(order.v1.astype(str)[order.v2.astype(str) == j])
                            try:
                                ln.remove(str(orderk.v2[i]))
                            except:
                                pass

                            for s in l2:
                                try:
                                    ln.remove(s)
                                except:
                                    pass
                            l2 = l2 + ln
                            lk = lk + ln
                    skip = False
                    for val in l1:
                        if val in l2:
                            skip = True
                            break
                    if skip == True:
                        continue
                    # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):  # ensure that there are no groups of variables that all have a connection with one another
                    #   continue
                    if len(lst) == len(set(lst)):
                        inde.append(
                            indexes[i]
                        )  # add used rows to inde (original row value)
                        order = pd.concat(
                            [order, orderk.loc[i].to_frame().T], ignore_index=True
                        )  # if conditions are met, add this to the dataframe of tree k
                    if (
                        len(order) == dimen - k
                    ):  # end loop when desired number of rows (number of variables - k) has been met
                        break
            orderk = order  # maker order = to tree k
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

        else:
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            v1_k = np.array(
                [v1_k[ind] for ind in orderk.index]
            ).T  # sort array of first node
            v2_k = np.array(
                [v2_k[ind] for ind in orderk.index]
            ).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
    if vine == "C":
        if len(orderk) > dimen - k:
            subnodes = np.unique(
                np.stack((np.array(orderk.v1), np.array(orderk.v2)))
            )  # see all the unique parent nodes
            taus = []  # list to put the sum of all tauabs in for a specific nodes
            for i in range(len(subnodes)):
                orderksub = orderk[
                    (orderk["v1"].apply(lambda x: x == subnodes[i]))
                    | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                ]
                taus.append(
                    sum(orderksub.tauabs)
                )  # calculate the sum of all taus for a specific variable
            i = np.where(np.array(taus) == max(taus))[0][
                0
            ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
            orderk = orderk[
                (orderk["v1"].apply(lambda x: x == subnodes[i]))
                | (orderk["v2"].apply(lambda x: x == subnodes[i]))
            ]  # select rows where this node is included
            orderk = orderk.sort_values(
                by="tauabs", ascending=False
            )  # sort this dataframe from highest to lowest tauabs
            inde = list(orderk.index)
            v1_k = np.array([v1_k[ind] for ind in inde]).T  # sort array of first node
            v2_k = np.array([v2_k[ind] for ind in inde]).T  # sort array of second node
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()
        else:
            orderk = orderk.sort_values(by="tauabs", ascending=False)
            v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
            v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
            orderk = orderk.reset_index(drop=True)
            orderk["tree"] = k - 1
            v1_2 = v1_k.copy()
            v2_2 = v2_k.copy()
            order2 = orderk.copy()

    for i in range(len(order2)):
        u3 = np.vstack(
            (v1_2[:, i], v2_2[:, i])
        ).T  # stacking the combination to fit copula to
        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator)  # fit the best copula
        aics.append(aic)  # add AIC to aics
        rhos.append(rho)  # add parameters to rhos
        cops.append(cop)  # add copula to cops
        coefs.append(coef)  # add coefficients to coefs
        logliks.append(loglik)  # add log-likelihood to logliks

    # add information to dataframe of tree k
    order2["rhos"] = rhos
    order2["cop"] = cops
    order2["AIC"] = aics
    order2["coefs"] = coefs
    order2["loglik"] = logliks

    # Check log-likelihood improvement for tree 2
    current_loglik = np.sum(logliks)
    ll_increase = np.sum(logliks)
    
    if ll_increase < min_ll_increase:
        early_stopped = True
        max_tree = 1
        if printing:
            print(f"Early stopping at tree 2: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
    else:
        prev_loglik = current_loglik + prev_loglik

    # loop through remaining trees if there are more than 3 variables and not early stopped
    if dimen > 3:
        for k in range(3, dimen):
            order = locals()[
                "order" + str(k - 1)
            ].copy()  # select the struture of the previous tree
            v1s = locals()[
                "v1_" + str(k - 1)
            ].copy()  # select the first nodes of the previous tree
            v2s = locals()[
                "v2_" + str(k - 1)
            ].copy()  # select the second nodes of the previous tree
            # create lists for the variables
            v1_k = []
            v2_k = []
            v1 = []
            v2 = []
            ktau = []
            rhos = []
            coefs = []
            cops = []
            node = []
            lk = []
            aics = []
            logliks = []
            rk = []
            for i in range(len(order) - 1):
                v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                copi = int(order.cop[i])  # copula of nodei
                pari = order.rhos[i]  # parameters of nodei
                for j in (
                    np.where(
                        np.array([item == v1i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v1i for item in list(order.v2[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v1[i + 1 :])])
                        | np.array([item == v2i for item in list(order.v2[i + 1 :])])
                    )[0]
                    + i
                    + 1
                ):  # see if possible connection between nodei and nodej
                    v1i = order.v1[i].copy()  # parent node 1 from nodei in tree
                    v2i = order.v2[i].copy()  # parent node 2 from nodei in tree
                    copj = int(order.cop[j])  # copula of nodei
                    parj = order.rhos[j]  # parameters of nodei
                    nodei = order.node[i]  # nodei
                    nodej = order.node[j]  # nodej
                    v1j = order.v1[j].copy()  # parent node 1 from nodej in tree
                    v2j = order.v2[j].copy()  # parent node 2 from nodej in tree
                    v1.append(nodei)  # new parent nodes 1
                    v2.append(nodej)  # new parent nodes 2
                    n = 2
                    ri = nodei[n + 1 :]  # select the values on the right side of node i
                    rj = nodej[n + 1 :]  # select the values on the right side of node i

                    if "g" in v1j:
                        v1j.remove("g")
                        v2j.remove("g")
                        v1i.remove("g")
                        v2i.remove("g")

                    # define left and right side of the node in tree k
                    if rj == ri:
                        if len(v1j) == 2:
                            lst = nodei[:n] + nodej[:n]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        else:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            for s in ri:
                                lst = [x for x in lst if x != s]
                            r3 = [max(set(lst), key=lst.count)]
                            li = [value for value in nodei[:n] if value not in r3]
                            lj = [value for value in nodej[:n] if value not in r3]
                            r = list(np.unique(ri + r3))
                        l = list(np.unique(li + lj))
                    else:
                        r = list(np.unique(ri + rj))
                        li = [value for value in nodei[:n] if value not in rj]
                        lj = [value for value in nodej[:n] if value not in ri]
                        if li == lj:
                            lst = v1i[:n] + v2i[:n] + v1j[:n] + v2j[:n]
                            lst = [x for x in lst if x != li[0]]
                            l3 = [min(set(lst), key=lst.count)]
                            l = list(np.unique(li + l3))
                        else:
                            l = list(np.unique(li + lj))
                    # select the parent node values
                    ui1 = v1s[:, i]
                    ui2 = v2s[:, i]
                    uj1 = v1s[:, j]
                    uj2 = v2s[:, j]
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni = 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1

                    # calculate the conditional CDF
                    v1igs = hfunc(copi, ui1, ui2, pari, un=uni, distribution=distribution)
                    v2jgs = hfunc(copj, uj1, uj2, parj, un=unj, distribution=distribution)
                    ktau.append(
                        abs(st.kendalltau(v1igs, v2jgs)[0])
                    )  # calculate the absolute kendall tau between v1igs and v2jgs, add it to the ktau list
                    # add the parent node values
                    v1_k.append(v1igs)
                    v2_k.append(v2jgs)

                    del uj1, ui1, ui2, uj2

                    node.append(l + ["g"] + r)  # set node name
                    lk.append(l)  # add left side of node
                    rk.append(r)  # add rigth side of node
            orderk = pd.DataFrame(
                {"v1": v1, "v2": v2, "tauabs": ktau, "node": node, "l": lk, "r": rk}
            )  # put information in dataframe for tree k

            if vine == "R" or vine == "D":
                if (
                    len(orderk) > dimen - k
                ):  # the tree will have the number of variables - k number of rows, if this is exceeded in orderk, a selection has to be made based on kendal tau
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    indexes = list(
                        orderk.index
                    )  # create a list the original order of the dataframe prior to sorting
                    orderk = orderk.reset_index(drop=True)  # reset the index
                    inde = (
                        []
                    )  #  list to put used rows of orderk in based on their oroginal position in the dataframe
                    inde2 = []  # list to put used rows of orderk i
                    for i in range(len(orderk)):
                        if i == 0:
                            order = orderk.head(
                                1
                            )  # loop through all the rows to include those pairs with the highest ktau first
                            inde.append(
                                indexes[i]
                            )  # add used rows to inde (original row value)
                            l = orderk.l[i]

                        else:
                            if (
                                orderk.v1[i] in list(orderk.v2[:i])
                                or orderk.v1[i] in list(orderk.v1[:i])
                            ) and (
                                orderk.v2[i] in list(orderk.v2[:i])
                                or orderk.v2[i] in list(orderk.v1[:i])
                            ):
                                continue
                            else:
                                inde.append(
                                    indexes[i]
                                )  # add used rows to inde (original row value)
                                inde2.append(i)  # add used rows to inde
                                order = pd.concat(
                                    [order, orderk.loc[i].to_frame().T],
                                    ignore_index=True,
                                )  # if conditions are met, add this to the dataframe of tree k
                                l = l + orderk.l[i]
                    if len(order) < (dimen - k):
                        for i in range(1, len(orderk)):
                            if i in inde2:
                                continue
                            lst = (
                                list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v1[i])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v1[i])
                                    ]
                                )
                                + list(
                                    order.v2.astype(str)[
                                        order.v1.astype(str) == str(orderk.v2[i])
                                    ]
                                )
                                + list(
                                    order.v1.astype(str)[
                                        order.v2.astype(str) == str(orderk.v2[i])
                                    ]
                                )
                            )
                            l1 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v1[i])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v1[i])
                                ]
                            )
                            lk = l1.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v1[i]))
                                    except:
                                        pass

                                    for s in l1:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l1 = l1 + ln
                                    lk = lk + ln

                            l2 = list(
                                order.v2.astype(str)[
                                    order.v1.astype(str) == str(orderk.v2[i])
                                ]
                            ) + list(
                                order.v1.astype(str)[
                                    order.v2.astype(str) == str(orderk.v2[i])
                                ]
                            )
                            lk = l2.copy()
                            while len(lk) > 0:
                                lk2 = lk.copy()
                                lk = []
                                for j in lk2:
                                    ln = list(
                                        order.v2.astype(str)[order.v1.astype(str) == j]
                                    ) + list(
                                        order.v1.astype(str)[order.v2.astype(str) == j]
                                    )
                                    try:
                                        ln.remove(str(orderk.v2[i]))
                                    except:
                                        pass

                                    for s in l2:
                                        try:
                                            ln.remove(s)
                                        except:
                                            pass
                                    l2 = l2 + ln
                                    lk = lk + ln
                            skip = False
                            for val in l1:
                                if val in l2:
                                    skip = True
                                    break
                            if skip == True:
                                continue
                            # if (len(l1) > 1 and len(l2) >0) or (len(l1) > 0 and len(l2) >1):
                            # continue
                            if len(lst) == len(set(lst)):
                                order = pd.concat(
                                    [order, orderk.loc[i].to_frame().T],
                                    ignore_index=True,
                                )  # if conditions are met, add this to the dataframe of tree k
                                inde.append(
                                    indexes[i]
                                )  # add used rows to inde (original row value)
                            if (
                                len(order) == dimen - k
                            ):  # end loop when desired number of rows (number of variables - k) has been met
                                break
                    orderk = order
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs

                    v1_k = np.array(
                        [v1_k[ind] for ind in inde]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in inde]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    
                    # Fit copulas for this tree
                    tree_aics = []
                    tree_logliks = []
                    tree_rhos = []
                    tree_cops = []
                    tree_coefs = []
                    
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator, early_stopped)  # fit the best copula
                        tree_aics.append(aic)  # add AIC to aics
                        tree_rhos.append(rho)  # add parameters to rhos
                        tree_cops.append(cop)  # add copula to cops
                        tree_coefs.append(coef)  # add coefficients to coefs
                        tree_logliks.append(loglik)  # add log-likelihood to logliks

                    # Check log-likelihood improvement
                   # Check log-likelihood improvement for tree 2
                        current_loglik = np.sum(logliks)
                        ll_increase = np.sum(logliks)
                        
                        if ll_increase < min_ll_increase:
                            early_stopped = True
                            max_tree = 1
                            if printing:
                                print(f"Early stopping at tree {k}: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
                        else:
                            prev_loglik = current_loglik + prev_loglik
                        # Add the fitted tree results
                        aics.extend(tree_aics)
                        rhos.extend(tree_rhos)
                        cops.extend(tree_cops)
                        coefs.extend(tree_coefs)
                        logliks.extend(tree_logliks)

                    # add information to dataframe of tree k
                    orderk["rhos"] = tree_rhos
                    orderk["cop"] = tree_cops
                    orderk["AIC"] = tree_aics
                    orderk["loglik"] = tree_logliks
                    orderk["coefs"] = tree_coefs
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

                else:
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    v1_k = np.array(
                        [v1_k[ind] for ind in orderk.index]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in orderk.index]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1

                    # Fit copulas for this tree
                    tree_aics = []
                    tree_logliks = []
                    tree_rhos = []
                    tree_cops = []
                    tree_coefs = []
                    
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator, early_stopped)  # fit the best copula
                        tree_aics.append(aic)  # add AIC to aics
                        tree_logliks.append(loglik)  # add log-likelihood to logliks
                        tree_rhos.append(rho)  # add parameters to rhos
                        tree_cops.append(cop)  # add copula to cops
                        tree_coefs.append(coef)  # add coefficients to coefs

                    # Check log-likelihood improvement
                    current_loglik = np.sum(logliks)
                    ll_increase = np.sum(logliks)
                    
                    if ll_increase < min_ll_increase:
                        early_stopped = True
                        max_tree = k - 1
                        if printing:
                            print(f"Early stopping at tree {k}: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
                    else:
                        prev_loglik = current_loglik + prev_loglik
                        # Add the fitted tree results
                        aics.extend(tree_aics)
                        rhos.extend(tree_rhos)
                        cops.extend(tree_cops)
                        coefs.extend(tree_coefs)
                        logliks.extend(tree_logliks)

                    # add information to dataframe of tree k
                    orderk["rhos"] = tree_rhos
                    orderk["cop"] = tree_cops
                    orderk["AIC"] = tree_aics
                    orderk["loglik"] = tree_logliks
                    orderk["coefs"] = tree_coefs
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

            if vine == "C":
                if len(orderk) > dimen - k:
                    subnodes = np.unique(
                        np.stack((np.array(orderk.v1), np.array(orderk.v2)))
                    )  # see all the unique parent nodes
                    taus = (
                        []
                    )  # list to put the sum of all tauabs in for a specific nodes
                    for i in range(len(subnodes)):
                        orderksub = orderk[
                            (orderk["v1"].apply(lambda x: x == subnodes[i]))
                            | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                        ]
                        taus.append(
                            sum(orderksub.tauabs)
                        )  # calculate the sum of all taus for a specific variable
                    i = np.where(np.array(taus) == max(taus))[0][
                        0
                    ]  # find where the sum of the taus is the highest to find the vairable to place in the center of the first tree
                    orderk = orderk[
                        (orderk["v1"].apply(lambda x: x == subnodes[i]))
                        | (orderk["v2"].apply(lambda x: x == subnodes[i]))
                    ]  # select rows where this node is included
                    orderk = orderk.sort_values(
                        by="tauabs", ascending=False
                    )  # sort this dataframe from highest to lowest tauabs
                    inde = list(orderk.index)
                    v1_k = np.array(
                        [v1_k[ind] for ind in inde]
                    ).T  # sort array of first node
                    v2_k = np.array(
                        [v2_k[ind] for ind in inde]
                    ).T  # sort array of second node
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1
                    v1_k = v1_k.copy()
                    v2_k = v2_k.copy()
                    
                    # Fit copulas for this tree
                    tree_aics = []
                    tree_logliks = []
                    tree_rhos = []
                    tree_cops = []
                    tree_coefs = []
                    
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator, early_stopped)  # fit the best copula
                        tree_aics.append(aic)  # add AIC to aics
                        tree_logliks.append(loglik)  # add log-likelihood to logliks
                        tree_rhos.append(rho)  # add parameters to rhos
                        tree_cops.append(cop)  # add copula to cops
                        tree_coefs.append(coef)  # add coefficients to coefs

                    # Check log-likelihood improvement
                    ll_increase = np.sum(logliks)
                    
                    if ll_increase < min_ll_increase:
                        early_stopped = True
                        max_tree = k - 1
                        if printing:
                            print(f"Early stopping at tree {k}: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
                    else:
                        prev_loglik = current_loglik + prev_loglik
                        # Add the fitted tree results
                        aics.extend(tree_aics)
                        logliks.extend(tree_logliks)
                        rhos.extend(tree_rhos)
                        cops.extend(tree_cops)
                        coefs.extend(tree_coefs)

                    # add information to dataframe of tree k
                    orderk["rhos"] = tree_rhos
                    orderk["cop"] = tree_cops
                    orderk["AIC"] = tree_aics
                    orderk["loglik"] = tree_logliks
                    orderk["coefs"] = tree_coefs
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

                else:
                    orderk = orderk.sort_values(by="tauabs", ascending=False)
                    v1_k = np.array([v1_k[ind] for ind in orderk.index]).T
                    v2_k = np.array([v2_k[ind] for ind in orderk.index]).T
                    orderk = orderk.reset_index(drop=True)
                    orderk["tree"] = k - 1

                    # Fit copulas for this tree
                    tree_aics = []
                    tree_rhos = []
                    tree_cops = []
                    tree_coefs = []
                    tree_logliks = []
                    
                    for j in range(len(orderk)):
                        u3 = np.vstack(
                            (v1_k[:, j], v2_k[:, j])
                        ).T  # stacking the combination to fit copula to
                        cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator, early_stopped)  # fit the best copula
                        tree_aics.append(aic)  # add AIC to aics
                        tree_logliks.append(loglik)  # add log-likelihood to logliks
                        tree_rhos.append(rho)  # add parameters to rhos
                        tree_cops.append(cop)  # add copula to cops
                        tree_coefs.append(coef)  # add coefficients to coefs

                    # Check log-likelihood improvement
                    ll_increase = np.sum(logliks)
                    
                    if ll_increase < min_ll_increase:
                        early_stopped = True
                        max_tree = k - 1
                        if printing:
                            print(f"Early stopping at tree {k}: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
                        
                    else:
                        prev_loglik = current_loglik + prev_loglik
                        # Add the fitted tree results
                        aics.extend(tree_aics)
                        logliks.extend(tree_logliks)    
                        rhos.extend(tree_rhos)
                        cops.extend(tree_cops)
                        coefs.extend(tree_coefs)

                    # add information to dataframe of tree k
                    orderk["rhos"] = tree_rhos
                    orderk["cop"] = tree_cops
                    orderk["AIC"] = tree_aics
                    orderk["loglik"] = tree_logliks
                    orderk["coefs"] = tree_coefs
                    locals()["v1_" + str(k)] = v1_k
                    locals()["v2_" + str(k)] = v2_k
                    locals()["order" + str(k)] = orderk

    order = pd.DataFrame(columns=order1.columns)  # dataframe to add all trees to
    for i in range(1, dimen):
        order = pd.concat([order, locals()["order" + str(i)]]).reset_index(
            drop=True
        )  # adding trees in dataframe
    print(order)
    # create array for the tree structure and copula matrices
    a = np.empty((dimen, dimen))
    c = np.empty((dimen, dimen))
    a[:] = np.nan
    c[:] = np.nan

    order["used"] = 0  # set used to 0 for nodes that have not been used
    for i in list(range(dimen-1))[
        ::-1
    ]:  # create the matix for the tree structure starting with the last tree
        k1 = sorted(
            np.array(
                order[(order.tree == i) & (order["used"] == 0)].node.iloc[0][:2]
            ).astype(int)
        )[::-1]
        order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
        t1 = i - 1
        ii = dimen - 2 - i
        a[i : dimen - ii, ii] = k1
        s = k1[-1]
        for j in list(range(0, i))[::-1]:  # continueing from tree i to the first tree
            orde = order[(order.tree == j) & (order["used"] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s, arr) == True:
                    inde = orde.iloc[k].name
                    a[j, ii] = arr[arr != s][0]
                    order["used"][inde] = 1

    a[0, dimen - 1] = a[0, dimen - 2]  # set first sample in sampling order
    orderk = pd.DataFrame(columns=order.columns)
    # create array for the copula parameter matrix
    p = np.empty((dimen, dimen))
    p[:] = np.nan
    p = p.astype(object)
    # fill array p with the parameters and c with the copulas, corresponding to the structure in c
    for i in list(range(dimen-1)):
        orde = order[order.tree == i]
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn, arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i, k] = orderj.rhos.iloc[0]
                    c[i, k] = orderj.cop.iloc[0]
                    if i == 0:
                        orderj.node.iloc[0] = list(akn)
                    else:
                        orderj.node.iloc[0] = (
                            list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1])
                        )
                    orderk = pd.concat([orderk, orderj]).reset_index(drop=True)

    # Set remaining copulas to independence (copula type 0) if early stopped
    #if early_stopped:
    #    for i in range(max_tree, dimen - 1):
    #        for k in range(dimen - 1 - i):
    #            c[i, k] = 0  # Independence copula
    #            p[i, k] = None  # No parameters for independence
    #    if printing:
    #        print(f"Set remaining copulas (trees {max_tree + 1} to {dimen - 1}) to independence")

    # print the copula structure if print == True
    # print the copula structure if print == True
    if printing == True:
        for i in list(range(0, dimen - 1)):
            orde = orderk[orderk.tree == i].reset_index(drop=True)
            print("** Tree: ", i + 1)
            for j in range(len(orde)):
                if i != 0:
                    nodej = ",".join(map(str, orde.node[j])).replace(",|,", "|")
                else:
                    nodej = ",".join(map(str, orde.node[j]))
                print(
                    nodej,
                    " ---> ",
                    copulas[int(orde.cop[j])],
                    ": parameters = ",
                    orde.coefs[j],
                )
        

    return a, p, c


# %% fitting vine copula with specific structure
def fit_vinecopstructure(u1, copsi, a, X, distribution=None, estimator =None, printing = True, min_ll_increase=1e-4):
    """
    Fit a regular vine copula to data based on a known vine structure matrix.

    Arguments:
        *u1* :  the data, provided as a numpy array where each column contains a seperate variable (eg. u1,u2,...,un), which have already been transferred to standard uniform margins (0<= u <= 1)

        *copsi* : A list of integers referring to the copulae of interest for which the fit has to be evauluated in the vine copula. eg. a list of [1, 10] refers to the Gaussian and Frank copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

    Returns:
     *p* : Parameters of the bivariate copulae provided as a triangular matrix.

     *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

    """

    dimen = a.shape[0]  # number of variables (number of columns)
    order = pd.DataFrame(
        columns=["node", "l", "r", "tree"]
    )  # dataframe for vinecopula information
    s = 0
    early_stopped = False
    for i in list(range(dimen - 1)):
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]  # Edge
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)  # Edge
            if i == 0:
                single_row_values = {
                    "node": list(akn),
                    "l": akn[0],
                    "r": akn[1],
                    "tree": i,
                }
            else:
                single_row_values = {
                    "node": list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1]),
                    "l": list(akn),
                    "r": list((ak.astype(int)[:i])[::-1]),
                    "tree": i,
                }

            order.loc[s] = single_row_values
            s = s + 1

    for t in list(range(dimen - 1)):  # loop through trees
        orderk = order[order.tree == t].reset_index(drop=True)  # select tre
        if t == 0:  # first tree
            orderk["v1"] = orderk.l
            orderk["v2"] = orderk.r
            rhos = []
            coefs = []
            v1_1 = []
            v2_1 = []
            cops = []
            tauabs = []
            logliks = []
            aics = []

            for j in range(len(orderk)):
                v1i = int(orderk.v1[j])  # first node
                v2i = int(orderk.v2[j])  # second node
                tauabs.append(
                abs(st.kendalltau(u1[:, v1i], u1[:, v2i])[0])
            ) 
                u3 = np.vstack(
                    (u1[:, v1i], u1[:, v2i])
                ).T  # stacking the combination to fit copula to
                cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator)  # fit the best copula
                rhos.append(rho)  # add parameters to rhos
                coefs.append(coef)  # add coefficients to coefs
                cops.append(cop)  # add copula to cops
                logliks.append(loglik)  # add log-likelihood to logliks
                v1_1.append(u1[:, v1i])  # add array of first node
                v2_1.append(u1[:, v2i])  # add array of second node
            v1_1 = np.array(v1_1).T  # v1
            v2_1 = np.array(v2_1).T  # v2

            # add information to dataframe of the first tree
            orderk["rhos"] = rhos
            orderk["coefs"] = coefs
            orderk["cop"] = cops
            orderk["tauabs"] = tauabs
            orderk["loglik"] = logliks
            orderk = orderk.sort_values(by="tauabs", ascending=False)
            v1_1 = v1_1[:, orderk.index]  # sort array of first node
            v2_1 =  v2_1[:, orderk.index] # sort array of second node
            orderk = orderk.reset_index(
        drop=True
    ) 
            
            
            
            
            

        else:

            v1k = []
            v2k = []
            tauabs = []
            for j in range(len(orderk)):
                orderk2 = locals()["order" + str(t)] # Define possible nodes of edge
                l = orderk.l[j] + orderk.r[j]
                subnodes = []
                for k in range(len(orderk2)):
                    subnodes.append(sum(1 for item in orderk2.node[k] if item in l))
                subnodes = np.array(subnodes) == len(l) - 1
                orderk2 = orderk2[subnodes].reset_index(drop=True)
                v1k.append(orderk2.node[0])  # node 1
                v2k.append(orderk2.node[1])  # node 2
            orderk["v1"] = v1k
            orderk["v2"] = v2k
            orderk2 = locals()["order" + str(t)].reset_index(drop=True)
            print(orderk2)
            v1s = locals()["v1_" + str(t)].copy()
            v2s = locals()["v2_" + str(t)].copy()
            v1_k = []
            v2_k = []
            rhos = []
            coefs = []
            cops = []
            logliks = []

            for k in range(len(orderk)):  # fitting copulas
                print(orderk)
                r = orderk.r[k]
                nodei = orderk.v1[k]  # nodei
                nodej = orderk.v2[k]  # nodej
                i = orderk2[orderk2["node"].apply(lambda x: x == nodei)].index[0]
                j = orderk2[orderk2["node"].apply(lambda x: x == nodej)].index[0]
                copj = int(orderk2.cop[j])  # copula of nodej
                parj = orderk2.rhos[j]  # parameters of nodej
                copi = int(orderk2.cop[i])  # copula of nodei
                pari = orderk2.rhos[i]  # parameters of nodei
                ri = orderk2.r[i]
                rj = orderk2.r[j]
                v1i = orderk2.v1[i]  # parent node 1 from nodei in tree
                v2i = orderk2.v2[i]
                v1j = orderk2.v1[j]  # parent node 1 from nodej in tree
                v2j = orderk2.v2[j]  # parent node 2 from nodej in tree
                ui1 = v1s[:, i]
                ui2 = v2s[:, i]
                uj1 = v1s[:, j]
                uj2 = v2s[:, j]
                if t > 1:
                    if "g" in v1j:
                        v1j.remove("g")
                        v2j.remove("g")
                        v1i.remove("g")
                        v2i.remove("g")
                    if set(r).issubset(set(v1i)):
                        uni = 1
                    elif set(r).issubset(set(v2i)):
                        uni = 2
                    elif set(rj).issubset(set(v2i[1:])):
                        uni = 2
                    elif set(rj).issubset(set(v1i[1:])):
                        uni = 1
                    if set(r).issubset(set(v1j)):
                        unj = 1
                    elif set(r).issubset(set(v2j)):
                        unj = 2
                    elif set(ri).issubset(set(v2j[1:])):
                        unj = 2
                    elif set(ri).issubset(set(v1j[1:])):
                        unj = 1

                else:
                    lst = orderk2.node[i] + orderk2.node[j]
                    s = max(set(lst), key=lst.count)
                    ui1 = v1_1[:, i]
                    ui2 = v2_1[:, i]
                    uj1 = v1_1[:, j]
                    uj2 = v2_1[:, j]
                    if v1i == s:
                        uni = 1
                        vi1 = v2i
                    else:
                        uni = 2
                        vi1 = v1i
                    if v1j == s:
                        unj = 1
                        vj1 = v2j
                    else:
                        unj = 2
                        vj1 = v1j


                # calculate the conditional CDF
                v1igs = hfunc(copi, ui1, ui2, pari, un=uni,distribution=distribution)
                v2jgs = hfunc(copj, uj1, uj2, parj, un=unj,distribution=distribution)
                u3 = np.vstack(
                    (v1igs, v2jgs)
                ).T  # stacking the combination to fit copula to
                cop, rho, aic, coef, loglik = bestcop(copsi, u3, X, distribution, estimator, early_stopped)  # fit the best copula
                
              
                rhos.append(rho)  # add parameters to rhos
                coefs.append(coef)
                cops.append(cop)  # add copula to cops
                logliks.append(loglik)  # add log-likelihood to logliks
                v1_k.append(v1igs)
                v2_k.append(v2jgs)

            ll_increase = np.sum(logliks)
            current_loglik = np.sum(logliks)
            if t ==1: 
                prev_loglik = current_loglik

            if ll_increase < min_ll_increase:
                early_stopped = True
                max_tree = k - 1
                if printing:
                    print(f"Early stopping at tree {k}: log-likelihood increase {ll_increase:.6f} < {min_ll_increase}")
            else:
                prev_loglik = current_loglik + prev_loglik

            orderk["rhos"] = rhos
            orderk["cop"] = cops
            orderk["loglik"] = logliks
            orderk["coefs"] = coefs
            locals()["v1_" + str(t + 1)] = np.array(v1_k).T
            locals()["v2_" + str(t + 1)] = np.array(v2_k).T

        locals()["order" + str(t + 1)] = orderk
    

    order = pd.DataFrame(columns=orderk.columns)  # dataframe to add all trees to
    for i in range(1, dimen):
        order = pd.concat([order, locals()["order" + str(i)]]).reset_index(
            drop=True
        )  # adding trees in dataframe
    # create array for the tree structure and copula matrices
    a = np.empty((dimen, dimen))
    c = np.empty((dimen, dimen))
    a[:] = np.nan
    c[:] = np.nan

    order["used"] = 0  # set used to 0 for nodes that have not been used
    for i in list(range(dimen-1))[
        ::-1
    ]:  # create the matix for the tree structure starting with the last tree
        k1 = sorted(
            np.array(
                order[(order.tree == i) & (order["used"] == 0)].node.iloc[0][:2]
            ).astype(int)
        )[::-1]
        order.loc[(order["tree"] == i) & (order["used"] == 0), "used"] = 1
        t1 = i - 1
        ii = dimen - 2 - i
        a[i : dimen - ii, ii] = k1
        s = k1[-1]
        for j in list(range(0, i))[::-1]:  # continueing from tree i to the first tree
            orde = order[(order.tree == j) & (order["used"] == 0)]
            for k in range(len(orde)):
                arr = np.array(orde.node.iloc[k][:2]).astype(int)
                if np.isin(s, arr) == True:
                    inde = orde.iloc[k].name
                    a[j, ii] = arr[arr != s][0]
                    order["used"][inde] = 1

    a[0, dimen - 1] = a[0, dimen - 2]  # set first sample in sampling order
    orderk = pd.DataFrame(columns=order.columns)
    # create array for the copula parameter matrix
    p = np.empty((dimen, dimen))
    p[:] = np.nan
    p = p.astype(object)
    # fill array p with the parameters and c with the copulas, corresponding to the structure in c
    for i in list(range(dimen-1)):
        orde = order[order.tree == i]
        for k in list(range(dimen - 1 - i)):
            ak = a[:, k]
            akn = np.array([ak[-1 - k], ak[i]]).astype(int)
            for j in range(len(orde)):
                arr = np.array(orde.node.iloc[j][:2]).astype(int)
                if sum(np.isin(akn, arr)) == 2:
                    orderj = order.loc[[orde.index[j]]]
                    p[i, k] = orderj.rhos.iloc[0]
                    c[i, k] = orderj.cop.iloc[0]
                    if i == 0:
                        orderj.node.iloc[0] = list(akn)
                    else:
                        orderj.node.iloc[0] = (
                            list(akn) + ["|"] + list((ak.astype(int)[:i])[::-1])
                        )
                    orderk = pd.concat([orderk, orderj]).reset_index(drop=True)

    

    # print the copula structure if print == True
    if printing == True:
        for i in list(range(0, dimen - 1)):
            orde = orderk[orderk.tree == i].reset_index(drop=True)
            print("** Tree: ", i + 1)
            for j in range(len(orde)):
                if i != 0:
                    nodej = ",".join(map(str, orde.node[j])).replace(",|,", "|")
                else:
                    nodej = ",".join(map(str, orde.node[j]))
                print(
                    nodej,
                    " ---> ",
                    copulas[int(orde.cop[j])],
                    ": parameters = ",
                    orde.coefs[j],
                )
        




    return p, c


def sample_vinecop(a, p, c, s, distribution = None):
    """
    Generate random samples from an R-vine.

    Arguments:
        *a* : The vine tree structure provided as a triangular matrix, composed of integers. The integer refers to different variables depending on which column the variable was in u1, where the first column is 0 and the second column is 1, etc.

        *p* : Parameters of the bivariate copulae provided as a triangular matrix.

        *c* : The types of the bivariate copulae provided as a triangular matrix, composed of integers referring to the copulae with the best fit. eg. a 1 refers to the gaussian copula  (see `Table 1 <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`__).

        *s* : number of samples to generate, provided as a positive scalar integer.



    Returns:
     *X2* :  the randomly sampled data data, provided as a numpy array where each column contains samples of a seperate variable (eg. u1,u2,...,un).

    """
    # Reference: Dißmann et al. 2013
    Ms = np.flipud(a)  # flip structure matrix
    P = np.flipud(p)  # flip parameter matrix
    C = np.flipud(c)  # flip copula matrix
    replace = {}  # dictionary for relabeling martix
    for i in range(int(max(np.unique(Ms)) + 1)):
        val = max(np.unique(Ms)) - i
        # Ms[k,k]
        replace.update({Ms[i, i]: val})  # relabel

    Ms = np.nan_to_num(Ms, nan=int(max(np.unique(Ms)) + 1))
    replace_func = np.vectorize(
        lambda x: replace.get(x, x)
    )  # Create a vectorized function for replacement
    M = replace_func(Ms)  # relabel
    M[M == np.max(Ms)] = np.nan
    Mm = M.copy()
    # max matrix
    for i in range(M.shape[0]):
        for k in range(M.shape[0]):
            if k == i:
                continue
            if i == 0:
                continue
            Mm[i, k] = max(Mm[i:, k])

    # Vdirect
    Vdir = np.empty((s, M.shape[0], M.shape[0]))
    Vdir[:] = np.nan
    # Vindirect
    Vindir = np.empty((s, M.shape[0], M.shape[0]))
    Vindir[:] = np.nan
    # Z2
    Z2 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    # Z1
    Z1 = np.empty((s, M.shape[0], M.shape[0]))
    Z2[:] = np.nan
    U = np.random.uniform(0, 1, (s, M.shape[0]))  # random uniform
    Vdir[:, -1, :] = U.copy()
    X = np.flip(U.copy(), 1)
    n = M.shape[0] - 1
    # sampling algorithm
    for k in range(n)[::-1]:
        for i in range(k + 1, n + 1):
            if M[i, k] == Mm[i, k]:
                Z2[:, i, k] = Vdir[:, i, int(n - Mm[i, k])]
            else:
                Z2[:, i, k] = Vindir[:, i, int(n - Mm[i, k])]
            Vdir[:, n, k] = hfuncinverse(
                int(C[i, k]), Z2[:, i, k], Vdir[:, n, k], P[i, k], un=2, distribution = distribution
            )
        X[:, int(n - k)] = Vdir[:, n, k]
        for i in range(k + 1, n + 1)[::-1]:
            Z1[:, i, k] = Vdir[:, i, k]
            Vdir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=2, distribution = distribution
            )
            Vindir[:, int(i - 1), k] = hfunc(
                int(C[i, k]), Z1[:, i, k], Z2[:, i, k], P[i, k], un=1, distribution = distribution
            )

    # Put X in the original order of the data
    replacedf = pd.DataFrame(list(replace.items()), columns=["Original", "Replacement"])
    replacedf = replacedf.sort_values(by="Original")
    X2 = np.array([])
    for i in replacedf.Replacement:
        if len(X2) == 0:
            X2 = X[:, int(i)].reshape(len(X), 1)
        else:
            X2 = np.hstack((X2, X[:, int(i)].reshape(len(X), 1)))
    return X2
