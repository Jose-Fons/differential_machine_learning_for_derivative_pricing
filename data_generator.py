# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:49:24 2023

@author: josef
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
#Option price function
def geometric_option(s, vol, K, r, T):
    n = vol.size()[1] #number of assets
    corr = torch.corrcoef(torch.transpose(s, 0, 1))
    
    #Total volatility
    sigma = torch.zeros(vol.size()[0])
    for i in range(n):
        for j in range(n):
            sigma += corr[i,j] * vol[:,i] * vol[:,j]
    sigma = torch.sqrt(sigma) / n
    
    #Sum of variances
    var_sum = torch.sum(vol*vol, 1)  / (2 * n)
    
    #Option parameters
    d2 = (torch.log(1/K) + (r - var_sum) * T) / (sigma * torch.sqrt(T))
    d1 = d2 + sigma * torch.sqrt(T)
    
    #Price
    dist = torch.distributions.normal.Normal(0,1)
    return torch.exp(-var_sum * T + sigma * sigma * T / 2) * dist.cdf(d1) - K * torch.exp(-r * T) * dist.cdf(d2) 


#Call price function
def call_price(s, vol, K, r, T):
    d1 = (torch.log(s/K) + (r + 0.5 * vol * vol) * T) / (vol * torch.sqrt(T))
    d2 = d1 - vol * torch.sqrt(T)
    dist = torch.distributions.normal.Normal(0,1)
    return s * dist.cdf(d1) - K * dist.cdf(d2) * torch.exp(- r * T)


#Option parameters
K = torch.tensor(4) #strike
r = torch.tensor(0.01) #interest rate
T = torch.tensor(10) #time to maturty (days?)
m = 10000 #number of simulations


#%%
#Datasets histograms
def plot_hist(x, x_sim, suptitle):
    n_inputs = x.size()[1]
    
    plt.figure()
    for i in range(n_inputs):
        plt.subplot(2, int(n_inputs/2), i+1)
        if i < int(n_inputs/2):
            title = 's' + str(i+1)
        else:
            title = 'vol' + str(i+1 - int(n_inputs/2))
        
        plt.hist(x[:,i], bins=50, density=True, color='blue', label='True')
        plt.hist(x_sim[:,i], bins=50, density=True, color='red', label='Simul.', alpha=0.5)
        plt.title(title)
        plt.legend()
        plt.show()
        
    plt.suptitle(suptitle)
    
    
    
#%%
#Real data
spots = pd.read_csv('Contenedor_Spot.txt', header=None)
spots.columns = ['DATE', 'NAME', 'SPOT']
volat = pd.read_csv('volatilidades.txt', sep='\t')

#Select prices and volatilites by name
price_bbva = spots.loc[spots.loc[:, 'NAME'] == 'EUR-BBV.MC', :]
price_caix = spots.loc[spots.loc[:, 'NAME'] == 'EUR-CRI.MC', :]
volat_bbva = volat.loc[volat.loc[:, 'NAME'] == 'EUR-BBV.MC', :]
volat_caix = volat.loc[volat.loc[:, 'NAME'] == 'EUR-CRI.MC', :]

#Convert all into tensors
torch.set_default_dtype(torch.float64)
price_bbva = torch.tensor(price_bbva['SPOT'].values)
price_caix = torch.tensor(price_caix['SPOT'].values)
volat_bbva = torch.tensor(volat_bbva['VOL(%)'].values/100)
volat_caix = torch.tensor(volat_caix['VOL(%)'].values/100)

#Concatenate prices and volatilities
s = torch.cat((price_bbva.view(len(price_bbva), -1).detach(), price_caix.view(len(price_caix), -1).detach()), 1)
vol = torch.cat((volat_bbva.view(len(volat_bbva), -1).detach(), volat_caix.view(len(volat_caix), -1).detach()), 1)

#Datasets
x = torch.cat((s, vol), 1).requires_grad_(True)
y = geometric_option(x[:,:int(x.size()[1]/2)], x[:,int(x.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x, y, z)
torch.save(dataset, './real_data.pt')


#%%
#Simulation data
s_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(price_bbva)).item(), torch.std(torch.log(price_bbva)).item(), m)).view(m, -1).detach(), 
                   torch.tensor(np.random.lognormal(torch.mean(torch.log(price_caix)).item(), torch.std(torch.log(price_caix)).item(), m)).view(m, -1).detach()), 1)
vol_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(volat_bbva)).item(), torch.std(torch.log(volat_bbva)).item(), m)).view(m, -1).detach(), 
                     torch.tensor(np.random.lognormal(torch.mean(torch.log(volat_caix)).item(), torch.std(torch.log(volat_caix)).item(), m)).view(m, -1).detach()), 1)

#Covariances
s_cov = torch.cov(torch.transpose(s, 0, 1)).detach().numpy()
s_sim_cov = torch.cov(torch.transpose(s_sim, 0, 1)).detach().numpy()
vol_cov = torch.cov(torch.transpose(vol, 0, 1)).detach().numpy()
vol_sim_cov = torch.cov(torch.transpose(vol_sim, 0, 1)).detach().numpy()

#Cholesky matrices
s_chol = torch.tensor(np.linalg.cholesky(s_cov))
s_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(s_sim_cov)))
vol_chol = torch.tensor(np.linalg.cholesky(vol_cov))
vol_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(vol_sim_cov)))

#Correlated simulations with identical distribution than real data
s_sim = torch.matmul(torch.matmul(s_sim, torch.transpose(s_sim_chol_inv, 0, 1)), torch.transpose(s_chol, 0, 1))
vol_sim = torch.matmul(torch.matmul(vol_sim, torch.transpose(vol_sim_chol_inv, 0, 1)), torch.transpose(vol_chol, 0, 1))
s_sim += torch.mean(s, 0, True) - torch.mean(s_sim, 0, True)
vol_sim += torch.mean(vol, 0, True) - torch.mean(vol_sim, 0, True)

#Datasets
x_sim = torch.cat((s_sim, vol_sim), 1).requires_grad_(True)
y = geometric_option(x_sim[:,:int(x_sim.size()[1]/2)], x_sim[:,int(x.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y = y.view(x_sim.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x_sim, y, z)
torch.save(dataset, './simulation_data.pt')


#%%
#Show histograms
# plot_hist(x, x_sim, 'Data Histograms (values)')


#%%
#Real data for regulation tests
s_dif = s[T.detach().numpy():, :] - s[:(-T.detach().numpy()), :] #T-period differences of values
vol_dif = vol[T.detach().numpy():, :] - vol[:(-T.detach().numpy()), :]
s_dif = torch.cat((torch.zeros(1, s_dif.size()[1]), s_dif), 0)
vol_dif = torch.cat((torch.zeros(1, vol_dif.size()[1]), vol_dif), 0)
s_regulation = s[0, :] + s_dif
vol_regulation = vol[0, :] + vol_dif

#Datasets
x = torch.cat((s_regulation, vol_regulation), 1).requires_grad_(True)
y = geometric_option(x[:,:int(x.size()[1]/2)], x[:,int(x.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x, y, z)
torch.save(dataset, './regulation_real_data.pt')


#%%
#Simulation data
s_regulation_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(s_regulation[:, 0])).item(), torch.std(torch.log(s_regulation[:, 0])).item(), m)).view(m, -1).detach(), 
                              torch.tensor(np.random.lognormal(torch.mean(torch.log(s_regulation[:, 1])).item(), torch.std(torch.log(s_regulation[:, 1])).item(), m)).view(m, -1).detach()), 1)
vol_regulation_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(vol_regulation[:, 0])).item(), torch.std(torch.log(vol_regulation[:, 0])).item(), m)).view(m, -1).detach(), 
                                torch.tensor(np.random.lognormal(torch.mean(torch.log(vol_regulation[:, 1])).item(), torch.std(torch.log(vol_regulation[:, 1])).item(), m)).view(m, -1).detach()), 1)

#Covariances
s_regulation_cov = torch.cov(torch.transpose(s_regulation, 0, 1)).detach().numpy()
s_regulation_sim_cov = torch.cov(torch.transpose(s_regulation_sim, 0, 1)).detach().numpy()
vol_regulation_cov = torch.cov(torch.transpose(vol_regulation, 0, 1)).detach().numpy()
vol_regulation_sim_cov = torch.cov(torch.transpose(vol_regulation_sim, 0, 1)).detach().numpy()

#Cholesky matrices
s_regulation_chol = torch.tensor(np.linalg.cholesky(s_regulation_cov))
s_regulation_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(s_regulation_sim_cov)))
vol_regulation_chol = torch.tensor(np.linalg.cholesky(vol_regulation_cov))
vol_regulation_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(vol_regulation_sim_cov)))

#Correlated simulations with identical distribution than real data for regulation
s_regulation_sim = torch.matmul(torch.matmul(s_regulation_sim, torch.transpose(s_regulation_sim_chol_inv, 0, 1)), torch.transpose(s_regulation_chol, 0, 1))
vol_regulation_sim = torch.matmul(torch.matmul(vol_regulation_sim, torch.transpose(vol_regulation_sim_chol_inv, 0, 1)), torch.transpose(vol_regulation_chol, 0, 1))
s_regulation_sim += torch.mean(s_regulation, 0, True) - torch.mean(s_regulation_sim, 0, True)
vol_regulation_sim += torch.mean(vol_regulation, 0, True) - torch.mean(vol_regulation_sim, 0, True)

#Datasets
x_sim = torch.cat((s_regulation_sim, vol_regulation_sim), 1).requires_grad_(True)
y = geometric_option(x_sim[:,:int(x.size()[1]/2)], x_sim[:,int(x.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y = y.view(x_sim.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x_sim, y, z)
torch.save(dataset, './regulation_simulation_data.pt')


#%%
#Show histograms
# plot_hist(x, x_sim, 'Data Histograms (profits)')

