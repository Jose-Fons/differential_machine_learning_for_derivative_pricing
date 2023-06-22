# -*- coding: utf-8 -*-
"""
DML vs. MC (geometric baket call)

@author: josef
"""

import torch
import time
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from itertools import product
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.set_default_dtype(torch.float64)

#%%
#Geometric basket call price function
def geom_price(s, vol, K, r, T, corr):
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

#Geometric basket call payoff function
def geom_payoff(s, K, T):
    n_assets = s.size()[1]
    prod = torch.prod(s[T:, :] / s[:(-T), :], 1)
    return torch.maximum(torch.pow(prod, 1/n_assets) - K, torch.tensor(0))

#Option parameters
K = torch.tensor(1.3) #strike
r = torch.tensor(0.01) #interest rate
T = torch.tensor(10) #time to maturty (days?)

#Simulation parameters
m = 100000 #number of simulations 
n_der = 1000 #number of extra initial values and number of derivatives
n_assets = 2
h = 1.01 #initial values step
s0 = torch.tensor([1.01, 1.015])
vol0 = torch.tensor([0.01, 0.005])
lower_vol0 = vol0
upper_vol0 = torch.tensor([0.3, 0.295])

#Compute greeks or not
greeks = True


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
s_real = torch.cat((price_bbva.view(len(price_bbva), -1).detach(), price_caix.view(len(price_caix), -1).detach()), 1)
vol_real = torch.cat((volat_bbva.view(len(volat_bbva), -1).detach(), volat_caix.view(len(volat_caix), -1).detach()), 1)

#Create more samples
s_real = torch.cat((s_real, s_real*h, s_real[100, :].view(-1, n_assets).detach()), 0)
vol_real = torch.cat((vol_real, vol_real*h, vol_real[100, :].view(-1, n_assets).detach()), 0)


#%%
#Time counters definition
time_simulation = 0
time_pricing = 0

#Price tensor definition
prices = torch.empty((1, n_der+1))

for i in range(n_der+1):
    #Start simulation time counter
    tic = time.perf_counter()
    
    #Geom. simulation data empty tensors
    s = torch.empty((m+1, n_assets))
    vol = torch.empty((m+1, n_assets))
    
    #Initialize tensors
    s[0, :] = s_real[i, :]
    vol[0, :] = vol_real[i, :]
    if i < int(n_der/2):
        lower_vol = lower_vol0
        upper_vol = upper_vol0
    else:
        lower_vol = lower_vol0 * h
        upper_vol = upper_vol0 * h
    
    #Volatility and underlying simulations
    vol[1:, :] = torch.rand((m, n_assets)) * (upper_vol - lower_vol) + lower_vol
    s[1:, :] = s[0, :] * torch.exp((r + vol[1:, :] * vol[1:, :] / 2) * T + vol[1:, :] * torch.sqrt(T) * torch.normal(0, 1, size=(m, n_assets)))

    #End simulation time counter
    toc = time.perf_counter()
    time_simulation += toc - tic
    
    #Start pricing time counter
    tic = time.perf_counter()
    
    #Valuation
    prices[0, i] = torch.exp(-r * T) * torch.mean(geom_payoff(s[1:, :], K, T), 0)
        
    #End pricing time counter
    toc = time.perf_counter()
    time_pricing += toc - tic
    
    if i % 100 == 0:
        print('Completed MC simulation bunches: ' + str(i) + '/' + str(n_der+1) + '\t(' + str(round(i/(n_der+1)*100, 2)) + '%)')
    
#Start greeks time counter
tic = time.perf_counter()
    
if greeks:
    #Greeks
    delta = (prices[0, 1:] - prices[0, :(-1)]).view(n_der, -1).detach() / (s0 * (h - 1))
    vega = (prices[0, 1:] - prices[0, :(-1)]).view(n_der, -1).detach() / (vol0 * (h - 1))
    MC_greeks = torch.cat((delta, vega), 1)

#End greeks time counter
toc = time.perf_counter()
time_greeks = toc - tic


#%%
# Lighting
torch.set_default_dtype(torch.float64)

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config, grad_pen, n_inputs=1, n_outputs=1, normalize=True):
        super(LitAutoEncoder, self).__init__()
        #Hyperparameters
        self.hidden_layers = config['hidden_layers']
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.alpha = config['alpha'] #ELU hyperparameter
        self.grad_pen = grad_pen #Loss function multiplier for gradients
        
        #Values for normalization
        self.epsilon = 1.0e-08 #to avoid singularities
        self.normalize = normalize
        
        #Neural Network architecture
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, self.hidden_layers[0], dtype=torch.float64),
            nn.ELU(self.alpha)
        )
        for layer, nodes in enumerate(self.hidden_layers[:-1]):
            self.layers.append(nn.Linear(self.hidden_layers[layer], 
                                            self.hidden_layers[layer + 1], 
                                            dtype=torch.float64))
            self.layers.append(nn.ELU(self.alpha))
        self.layers.append(nn.Linear(self.hidden_layers[-1], n_outputs, 
                                        dtype=torch.float64))


    def forward(self, x):
        stats = self.trainer.datamodule.stats
        if self.normalize:
            #Normalization
            x = (x - stats['x']['mean']) / (stats['x']['std'] + self.epsilon)
            
        #Forward
        torch.set_grad_enabled(True)
        x = x.requires_grad_(True)
        y_pred = self.layers(x)
        
        #Backward
        z_pred = torch.autograd.grad(y_pred.sum(), x, retain_graph=True)[0]
        
        if self.normalize:
            #De-normalization
            y_pred = stats['y']['mean'] + (stats['y']['std'] + self.epsilon) * y_pred
            z_pred = z_pred * stats['y']['std'] / stats['x']['std']
        
        return y_pred, z_pred
    
    
    def steps_loop(self, x, y, z):
        #Normalization
        stats = self.trainer.datamodule.stats
        if self.normalize:
            stats['z']['norm'] = stats['z']['norm'] * stats['x']['std'] / stats['y']['std']
            y = (y - stats['y']['mean']) / (stats['y']['std'] + self.epsilon)
            z = z * stats['x']['std'] / stats['y']['std']
        
        #Prediction
        y_pred, z_pred = self(x)
        if normalize:
            #Re-normalization
            y_pred = (y_pred - stats['y']['mean']) / (stats['y']['std'] + self.epsilon)
            z_pred = z_pred * stats['x']['std'] / stats['y']['std']
        
        #Loss
        loss = nn.MSELoss()(y_pred, y) + self.grad_pen * torch.mean(torch.pow((z_pred - z) / stats['z']['norm'], 2), 0).sum()
        return loss
    
    
    def training_step(self, train_batch, batch_idx):
        x, y, z = train_batch
        loss = self.steps_loop(x, y, z)
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y, z = val_batch
        loss = self.steps_loop(x, y, z)
        if tune.is_session_enabled():
            tune.report(loss=loss.detach().numpy())
        self.log('val_loss', loss)
        return loss
    
    
    def test_step(self, test_batch, batch_idx):
        x, y, z = test_batch
        loss = self.steps_loop(x, y, z)
        self.log('test_loss', loss)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ----------------------------------------
#Data management
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.stats = {}

    def setup(self, stage):
        self.train, self.val, self.test = torch.utils.data.random_split(self.dataset, [0.7, 0.2, 0.1])

        #Statistical values
        self.stats = {'x': {'mean': torch.mean(self.train[:][0], 0, True), 'std': torch.std(self.train[:][0], 0, True)}, 
                      'y': {'mean': torch.mean(self.train[:][1]), 'std': torch.std(self.train[:][1])}, 
                      'z': {'norm': torch.linalg.vector_norm(self.train[:][2], dim=0, keepdim=True)}}

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


#%%
#Test data
corr = torch.corrcoef(torch.transpose(s_real, 0, 1))
x = torch.cat((s_real, vol_real), 1).requires_grad_(True)
y = geom_price(x[:, :n_assets], x[:, n_assets:], K, r, T, corr)
z = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
x = x.requires_grad_(False)
y = y.view(x.size()[0], -1).detach()
dataset_real = torch.utils.data.TensorDataset(x, y, z)

# ---------------------

#Simulation data
s_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(price_bbva)).item(), torch.std(torch.log(price_bbva)).item(), m)).view(m, -1).detach(), 
                   torch.tensor(np.random.lognormal(torch.mean(torch.log(price_caix)).item(), torch.std(torch.log(price_caix)).item(), m)).view(m, -1).detach()), 1)
vol_sim = torch.cat((torch.tensor(np.random.lognormal(torch.mean(torch.log(volat_bbva)).item(), torch.std(torch.log(volat_bbva)).item(), m)).view(m, -1).detach(), 
                     torch.tensor(np.random.lognormal(torch.mean(torch.log(volat_caix)).item(), torch.std(torch.log(volat_caix)).item(), m)).view(m, -1).detach()), 1)

#Covariances
s_cov = torch.cov(torch.transpose(s_real, 0, 1)).detach().numpy()
s_sim_cov = torch.cov(torch.transpose(s_sim, 0, 1)).detach().numpy()
vol_cov = torch.cov(torch.transpose(vol_real, 0, 1)).detach().numpy()
vol_sim_cov = torch.cov(torch.transpose(vol_sim, 0, 1)).detach().numpy()

#Cholesky matrices
s_chol = torch.tensor(np.linalg.cholesky(s_cov))
s_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(s_sim_cov)))
vol_chol = torch.tensor(np.linalg.cholesky(vol_cov))
vol_sim_chol_inv = torch.tensor(np.linalg.inv(np.linalg.cholesky(vol_sim_cov)))

#Correlated simulations with identical distribution than real data
s_sim = torch.matmul(torch.matmul(s_sim, torch.transpose(s_sim_chol_inv, 0, 1)), torch.transpose(s_chol, 0, 1))
vol_sim = torch.matmul(torch.matmul(vol_sim, torch.transpose(vol_sim_chol_inv, 0, 1)), torch.transpose(vol_chol, 0, 1))
s_sim += torch.mean(s_real, 0, True) - torch.mean(s_sim, 0, True)
vol_sim += torch.mean(vol_real, 0, True) - torch.mean(vol_sim, 0, True)

corr = torch.corrcoef(torch.transpose(s_sim, 0, 1))

#Datasets
x_sim = torch.cat((s_sim, vol_sim), 1).requires_grad_(True)
y_sim = geom_price(x_sim[:, :n_assets], x_sim[:, n_assets:], K, r, T, corr)
z_sim = torch.autograd.grad(y_sim.sum(), x_sim, retain_graph=True)[0]
x_sim = x_sim.requires_grad_(False)
y_sim = y_sim.view(x_sim.size()[0], -1).detach()
dataset_sim = torch.utils.data.TensorDataset(x_sim, y_sim, z_sim)


#%%
#Start time counter
tic = time.perf_counter()

# Hyperparameter tuning
epochs = 25
normalize = True
grad_pen = greeks * 10 + (not greeks) * 0

config = {
    'hidden_layers': tune.choice([layer for r in [2,3] for layer in product([16, 32, 64, 128, 256], repeat=r)]),
    # 'hidden_layers': tune.choice([layer for r in [1,3,5] for layer in product([32, 64, 128], repeat=r)]),
    # 'hidden_layers': tune.choice([tuple([nodes] * layers) for nodes in [16, 32, 64, 128, 256] for layers in range(1,6)]),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([32, 64, 128]),
    'alpha': tune.uniform(0, 10),
}

def train_tune(config, epochs=25):
    data = DataModule(dataset_sim, config)
    model = LitAutoEncoder(config, grad_pen, n_inputs=dataset_sim[:][0].size()[1], n_outputs=1, normalize=normalize)
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer.fit(model, data)

scheduler = ASHAScheduler(max_t=epochs, grace_period=1, reduction_factor=2)
reporter = CLIReporter(parameter_columns=['hidden_layers', 'learning_rate', 
                                          'batch_size', 'alpha'], 
                        metric_columns=['loss', 'training_iteration'])

result = tune.run(train_tune, config=config, progress_reporter=reporter, 
                  scheduler=scheduler, num_samples=200, metric='loss', mode='min', 
                  raise_on_failed_trial=False)

#End time counter
toc = time.perf_counter()
time_tune = toc - tic


#%%
#Start time counter
tic = time.perf_counter()

# Train with best hyperparameters
epochs = 25
normalize = True
grad_pen = greeks * 10 + (not greeks) * 0
config = {'hidden_layers': (128, 256, 64, 16), 'learning_rate': 0.0020434855190057453, 'batch_size': 64, 'alpha': 1.147389576148048}
# loss=8.61778564435248e-05

data = DataModule(dataset_sim, config)
model = LitAutoEncoder(config, grad_pen, n_inputs=dataset_sim[:][0].size()[1], n_outputs=1, normalize=normalize)
trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
trainer.fit(model, data)

#End time counter
toc = time.perf_counter()
time_train = toc - tic


#%%
#Start time counter
tic = time.perf_counter()

#Payoff and gradient prediction
y_pred, z_pred = model(dataset_real[:][0])

#Start time counter
toc = time.perf_counter()
time_predict = toc - tic


#%%
#Errors MC
error_mc_price = (torch.transpose(prices, 0, 1) - dataset_real[:][1]) / dataset_real[:][1]
error_mc_price_mean = torch.mean(error_mc_price)
error_mc_price_std = torch.std(error_mc_price)
if greeks:
    error_mc_greeks = MC_greeks - dataset_real[1:][2]
    error_mc_greeks_mean = torch.mean(error_mc_greeks)
    error_mc_greeks_std = torch.std(error_mc_greeks)

#Error DML
error_dml_price = (y_pred - dataset_real[:][1]) / dataset_real[:][1]
error_dml_price_mean = torch.mean(error_dml_price)
error_dml_price_std = torch.std(error_dml_price)
if greeks:
    error_dml_greeks = z_pred - dataset_real[:][2]
    error_dml_greeks_mean = torch.mean(error_dml_greeks)
    error_dml_greeks_std = torch.std(error_dml_greeks)


#%%
print('\n\nComputation times:')
print('--------------------')
print('Monte-Carlo:')
print('\tSimulations:\t\t\t\t' + str(round(time_simulation, 3)) + ' s')
print('\tPricing:\t\t\t\t\t' + str(round(time_pricing, 3)) + ' s')
print('\tDifferentiation:\t\t\t' + str(round(time_greeks, 3)) + ' s')
print('-----')
print('Differential Machine Learning:')
print('\tHyperparameter tuning:\t\t' + str(round(time_tune, 3)) + ' s')
print('\tTraining process:\t\t\t' + str(round(time_train, 3)) + ' s')
print('\tPrediction:\t\t\t\t\t' + str(round(time_predict, 3)) + ' s')

print('\n\nComputation errors:')
print('--------------------')
print('Monte-Carlo:')
print('\tPrice error mean:\t\t\t' + str(100*error_mc_price_mean.item()) + ' %')
print('\tPrice error std:\t\t\t' + str(100*error_mc_price_std.item()) + ' %')
if greeks:
    print('\tGreeks error mean:\t\t\t' + str(100*error_mc_greeks_mean.item()) + ' %')
    print('\tGreeks error std:\t\t\t' + str(100*error_mc_greeks_std.item()) + ' %')
print('-----')
print('Differential Machine Learning:')
print('\tPrice error mean:\t\t\t' + str(100*error_dml_price_mean.item()) + ' %')
print('\tPrice error std:\t\t\t' + str(100*error_dml_price_std.item()) + ' %')
if greeks:
    print('\tGreeks error mean:\t\t\t' + str(100*error_dml_greeks_mean.item()) + ' %')
    print('\tGreeks error std:\t\t\t' + str(100*error_dml_greeks_std.item()) + ' %')