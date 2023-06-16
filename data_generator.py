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
def geometric_option(s, vol, K, r, T, corr):
    n = vol.size()[1] #number of assets
    # corr = torch.corrcoef(torch.transpose(s, 0, 1))
    
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
K = torch.tensor(1) #strike
r = torch.tensor(0.01) #interest rate
T = torch.tensor(10) #time to maturty (days?)
m = 100000 #number of simulations


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
#Distributions plot
def plot_distribution(x, x_sim, x_dif=None, suptitle='Distributions pair-plot'):
    n_inputs = x.size()[1]
    n_assets = int(n_inputs / 2)
    
    plt.figure(figsize=(3*n_inputs, 3*n_inputs))
    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.975, top=0.95, wspace=0.3, hspace=0.2)
    
    for i in range(n_inputs):
        for j in range(n_inputs):
            plt.subplot(n_inputs, n_inputs, i*n_inputs + j+1)
            
            #Plots
            if i == j:
                plt.hist(x_sim[:,i], bins=50, density=True, label='Simul.')
                plt.hist(x[:,i], bins=50, density=True, color='orange', label='Real dif.', alpha=0.6)
                plt.legend()
            else:
                plt.plot(x_sim[:,j], x_sim[:,i], '.', label='Simul.')
                if x_dif != None:
                    plt.plot(x_dif[:,j], x_dif[:,i], '.', color='orange', label='Real dif.')
                    plt.legend()
              
            #Labels
            if i == (n_inputs - 1):
                if j < n_assets:
                    plt.xlabel('s' + str(j+1))
                else:
                    plt.xlabel('vol' + str(j+1 - n_assets))
                    
            if j == 0:
                if i < n_assets:
                    plt.ylabel('s' + str(i+1))
                else:
                    plt.ylabel('vol' + str(i+1 - n_assets))
                    
    #Title
    plt.suptitle(suptitle)
    plt.show()
    


#%%
#Real data
spots = pd.read_csv('./Data/Contenedor_Spot.txt', header=None)
spots.columns = ['DATE', 'NAME', 'SPOT']
volat = pd.read_csv('./Data/volatilidades.txt', sep='\t')

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

corr = torch.corrcoef(torch.transpose(s, 0, 1))

#Datasets
x = torch.cat((s, vol), 1).requires_grad_(True)
y = geometric_option(x[:,:int(x.size()[1]/2)], x[:,int(x.size()[1]/2):], K, r, T, corr)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x, y, z)
torch.save(dataset, './Data/geometric_real_data.pt')


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

corr = torch.corrcoef(torch.transpose(s_sim, 0, 1))

#Datasets
x_sim = torch.cat((s_sim, vol_sim), 1).requires_grad_(True)
y = geometric_option(x_sim[:,:int(x_sim.size()[1]/2)], x_sim[:,int(x_sim.size()[1]/2):], K, r, T, corr)
z = torch.autograd.grad(y.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y = y.view(x_sim.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x_sim, y, z)
torch.save(dataset, './Data/geometric_simulation_data.pt')


#%%
#Show histograms
# plot_hist(x, x_sim, 'Data Histograms (values)')


#%%
#Real data for regulation tests
s_dif = s[T.detach().numpy():, :] - s[:(-T.detach().numpy()), :] #T-period differences of values
vol_dif = vol[T.detach().numpy():, :] - vol[:(-T.detach().numpy()), :]
s_dif = torch.cat((torch.zeros(1, s_dif.size()[1]), s_dif), 0)
vol_dif = torch.cat((torch.zeros(1, vol_dif.size()[1]), vol_dif), 0)
s_reg = s[0, :] + s_dif
vol_reg = vol[0, :] + vol_dif

corr = torch.corrcoef(torch.transpose(s_dif, 0, 1))

#Datasets
x = torch.cat((s_reg, vol_reg), 1).requires_grad_(True)
y = geometric_option(x[:,:int(x.size()[1]/2)], x[:,int(x.size()[1]/2):], K, r, T, corr)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x, y, z)
torch.save(dataset, './Data/regulation_real_data.pt')


#%%
#Simulation data for regulation tests
s_reg_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(s_reg[:, 0])).item(), torch.std(torch.log(s_reg[:, 0])).item(), m)).view(m, -1).detach(), 
                       torch.tensor(np.random.lognormal(torch.mean(torch.log(s_reg[:, 1])).item(), torch.std(torch.log(s_reg[:, 1])).item(), m)).view(m, -1).detach()), 1)
vol_reg_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(vol_reg[:, 0])).item(), torch.std(torch.log(vol_reg[:, 0])).item(), m)).view(m, -1).detach(), 
                         torch.tensor(np.random.lognormal(torch.mean(torch.log(vol_reg[:, 1])).item(), torch.std(torch.log(vol_reg[:, 1])).item(), m)).view(m, -1).detach()), 1)

#Covariances
s_reg_cov = torch.cov(torch.transpose(s_reg, 0, 1)).detach().numpy()
s_reg_sim_cov = torch.cov(torch.transpose(s_reg_sim, 0, 1)).detach().numpy()
vol_reg_cov = torch.cov(torch.transpose(vol_reg, 0, 1)).detach().numpy()
vol_reg_sim_cov = torch.cov(torch.transpose(vol_reg_sim, 0, 1)).detach().numpy()

#Cholesky matrices
s_reg_chol = torch.tensor(np.linalg.cholesky(s_reg_cov))
s_reg_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(s_reg_sim_cov)))
vol_reg_chol = torch.tensor(np.linalg.cholesky(vol_reg_cov))
vol_reg_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(vol_reg_sim_cov)))

#Correlated simulations with identical distribution than real data for regulation
s_reg_sim = torch.matmul(torch.matmul(s_reg_sim, torch.transpose(s_reg_sim_chol_inv, 0, 1)), torch.transpose(s_reg_chol, 0, 1))
vol_reg_sim = torch.matmul(torch.matmul(vol_reg_sim, torch.transpose(vol_reg_sim_chol_inv, 0, 1)), torch.transpose(vol_reg_chol, 0, 1))
s_reg_sim += torch.mean(s_reg, 0, True) - torch.mean(s_reg_sim, 0, True)
vol_reg_sim += torch.mean(vol_reg, 0, True) - torch.mean(vol_reg_sim, 0, True)

corr = torch.corrcoef(torch.transpose(s_reg_sim, 0, 1))

#Datasets
x_sim = torch.cat((s_reg_sim, vol_reg_sim), 1).requires_grad_(True)
y = geometric_option(x_sim[:,:int(x_sim.size()[1]/2)], x_sim[:,int(x_sim.size()[1]/2):], K, r, T, corr)
z = torch.autograd.grad(y.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y = y.view(x_sim.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x_sim, y, z)
torch.save(dataset, './Data/regulation_simulation_data.pt')


#%%
#Show distributions
# x_dif = torch.cat((s_dif, vol_dif), 1)
# x_reg = torch.cat((s_reg, vol_reg), 1)
# plot_distribution(x, x_sim, x_reg , 'Distributions pair-plot\n(possible scenarios based on differences)')


#%%
#Call real data
lower_s = 0.01
upper_s = 2
lower_v = 0.01
upper_v = 0.5
K = torch.tensor(1)
r = torch.tensor(0.01)
T = torch.tensor(2)
s_call = torch.linspace(lower_s, upper_s, m)
vol_call = torch.linspace(lower_v, upper_v, m)

#Datasets
x = torch.cat((s_call.view(m, -1).detach(), vol_call.view(m, -1).detach()), 1).requires_grad_(True)
y = call_price(x[:,:int(x.size()[1]/2)], x[:,int(x.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x, y, z)
torch.save(dataset, './Data/call_real_data.pt')


#%%
#Call simulation data
s_call_sim = torch.tensor(np.random.uniform(lower_s, upper_s, m))
vol_call_sim = torch.tensor(np.random.uniform(lower_v, upper_v, m))

#Datasets
x_sim = torch.cat((s_call_sim.view(m, -1).detach(), vol_call_sim.view(m, -1).detach()), 1).requires_grad_(True)
y = call_price(x_sim[:,:int(x_sim.size()[1]/2)], x_sim[:,int(x_sim.size()[1]/2):], K, r, T)
z = torch.autograd.grad(y.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y = y.view(x_sim.size()[0], -1).detach()

dataset = torch.utils.data.TensorDataset(x_sim, y, z)
torch.save(dataset, './Data/call_simulation_data.pt')