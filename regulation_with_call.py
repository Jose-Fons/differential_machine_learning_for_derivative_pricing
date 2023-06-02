# -*- coding: utf-8 -*-
"""
P&L

@author: Jose
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from torchmetrics import SpearmanCorrCoef

#%%
#Basilea tests function
def basilea_test(HPL, RTPL, title, cumulative=True):
    #Spearman Correlation Coefficient
    spear_test = SpearmanCorrCoef()(HPL[:, 0], RTPL[:, 0]).item()
    
    if spear_test <= 0.7:
        spear_color = 'red'
    elif spear_test <= 0.8:
        spear_color = 'amber'
    else:
        spear_color = 'green'
    
    #Kolmogorov-Smirnov test
    threshold = np.linspace(min(min(HPL), min(RTPL)).item(), max(max(HPL), max(RTPL)).item(), 10000)
    n_observ_HPL = np.empty_like(threshold)
    n_observ_RTPL = np.empty_like(threshold)
    for i, PL in enumerate(threshold):
        n_observ_HPL[i] = torch.sum(1 * (HPL < PL)).item()
        n_observ_RTPL[i] = torch.sum(1 * (RTPL < PL)).item()
    ks_test = max(abs(0.004 * (n_observ_RTPL - n_observ_HPL)))
    
    if ks_test >= 0.12:
        ks_color = 'red'
    elif ks_test >= 0.09:
        ks_color = 'orange'
    else:
        ks_color = 'green'
    
    #Histogram
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(HPL[:, 0], bins=50, color='blue', cumulative=cumulative, label='Groun Truth')
    ax.hist(RTPL[:, 0].detach().numpy(), bins=50, color='red', cumulative=cumulative, label='Prediction', alpha=0.5)
    if cumulative:
        ax.set_ylabel('Cumulated Counts')
    else:
        ax.set_ylabel('Counts')
    ax.set_title('Profit & Loss \n (base scenario for predicted P&L: ' + title + ')')
    ax.legend(loc='lower left')
    
    ax.text(0, 0.99, 'Spearman: ' + str(round(spear_test, 5)), color=spear_color, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    ax.text(0, 0.95, 'K-S: ' + str(round(ks_test, 5)), color=ks_color, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    
    plt.show()

#%%
#Call price function
def call_price(s, vol, K, r, T):
    d1 = (torch.log(s/K) + (r + 0.5 * vol * vol) * T) / (vol * torch.sqrt(T))
    d2 = d1 - vol * torch.sqrt(T)
    dist = torch.distributions.normal.Normal(0,1)
    return s * dist.cdf(d1) - K * dist.cdf(d2) * torch.exp(- r * T)

#%%
#Loading data
torch.set_default_dtype(torch.float64)
real = torch.load('./real_data.pt')
pred = torch.load('./predicted_data.pt')

#Call parameters
K = torch.tensor(4) #strike
r = torch.tensor(0.01) #interest rate
T = torch.tensor(10) #time to maturty (days?)

#Portfolios
call_prices = call_price(real[:][0][:, 0], real[:][0][:, 2], K, r, T)
call_prices = call_prices.view(real[:][0][:, 0].size()[0], -1).detach()
w_real = (real[:][1][1] - real[:][1][0]) / (call_prices[1] - call_prices[0]) #portfolio weights
w_pred = (pred[:][1][1] - pred[:][1][0]) / (call_prices[1] - call_prices[0])
portfolio_real = real[:][1] + w_real * call_prices
portfolio_pred = pred[:][1] + w_pred * call_prices

#Profit & Loss
real_PL = portfolio_real[1:] - portfolio_real[0]
pred_PL1 = portfolio_pred[1:] - portfolio_real[0]
pred_PL2 = portfolio_pred[1:] - portfolio_pred[0]

#%%
#Results
basilea_test(real_PL, pred_PL1, 'Ground Truth', cumulative=True)
basilea_test(real_PL, pred_PL2, 'Prediction', cumulative=True)