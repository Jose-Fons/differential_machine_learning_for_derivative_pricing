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
def basilea_test(HPL, RTPL, title, ax, n_axis, cumulative=True,):
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
    ax[n_axis].hist(HPL[:, 0], bins=50, color='blue', cumulative=cumulative, density=True, label='Groun Truth')
    ax[n_axis].hist(RTPL[:, 0].detach().numpy(), bins=50, color='red', cumulative=cumulative, density=True, label='Prediction', alpha=0.5)
    ax[n_axis].set_xlim(min(min(HPL), min(RTPL)).item(), max(max(HPL), max(RTPL)).item())
    if cumulative:
        ax[n_axis].set_ylabel('Cumulated Counts')
    else:
        ax[n_axis].set_ylabel('Counts')
    ax[n_axis].set_title('(base scenario for predicted P&L: ' + title + ')')
    ax[n_axis].legend(loc='center left')
    
    ax[n_axis].text(0, 0.99, 'Spearman: ' + str(round(spear_test, 5)), color=spear_color, verticalalignment='top', horizontalalignment='left', transform=ax[n_axis].transAxes)
    ax[n_axis].text(0, 0.95, 'K-S: ' + str(round(ks_test, 5)), color=ks_color, verticalalignment='top', horizontalalignment='left', transform=ax[n_axis].transAxes)
    

#%%
#Loading data
torch.set_default_dtype(torch.float64)
real = torch.load('./Data/regulation_real_data.pt')
pred = torch.load('./Data/predicted_data.pt')

#Profit & Loss
real_PL = real[1:][1] - real[0][1]
pred_PL1 = pred[1:][1] - real[0][1]
pred_PL2 = pred[1:][1] - pred[0][1]

#%%
#Results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.4)

basilea_test(real_PL, pred_PL1, 'Ground Truth', ax, 0, cumulative=True)
basilea_test(real_PL, pred_PL2, 'Prediction', ax, 1, cumulative=True)

plt.suptitle('Profit & Loss')
plt.show()