# -*- coding: utf-8 -*-
"""
Differential Machine Learning for Derivative Pricing

@author: josef
"""

#%%
# Libraries
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
        self.grad_pen = grad_pen #Loss function multiplier for differential
        
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
# Kolmogorov-Smirnov test
def ks_test(HPL, RTPL):
    threshold = np.linspace(min(min(HPL), min(RTPL)).item(), max(max(HPL), max(RTPL)).item(), 10000)
    n_observ_HPL = np.empty_like(threshold)
    n_observ_RTPL = np.empty_like(threshold)
    for i, PL in enumerate(threshold):
        n_observ_HPL[i] = torch.sum(1 * (HPL < PL)).item()
        n_observ_RTPL[i] = torch.sum(1 * (RTPL < PL)).item()
        
    return max(abs(0.004 * (n_observ_RTPL - n_observ_HPL)))


#%%
#Hyperparameter configurations
epochs = 25
normalize = True
# config = []
# for i, layers in enumerate([1, 3, 4, 5]):
#     for j, nodes in enumerate([32, 64, 128]):
#         config.append({'hidden_layers': tuple([nodes] * layers), 
#                        'learning_rate': 0.0015, 'batch_size': 64, 'alpha': 0.7})
config = [{'hidden_layers': (64, 256, 32), 'learning_rate': 0.0009976629846684705, 'batch_size': 64, 'alpha': 0.11631924590986942},
          {'hidden_layers': (128, 256, 64, 16), 'learning_rate': 0.0020434855190057453, 'batch_size': 64, 'alpha': 1.147389576148048},
          {'hidden_layers': (256, 64, 64), 'learning_rate': 0.003242405561782988, 'batch_size': 64, 'alpha': 0.7843895203261764},
          {'hidden_layers': (16, 16, 64), 'learning_rate': 0.0012477005179843312, 'batch_size': 128, 'alpha': 1.887009372781434},
          {'hidden_layers': (256, 128, 64, 32), 'learning_rate': 0.026235662270740853, 'batch_size': 64, 'alpha': 1.8200700322792285},
          {'hidden_layers': (256, 128, 64), 'learning_rate': 0.0002614783398977329, 'batch_size': 32, 'alpha': 1.159029241344316},
          {'hidden_layers': (64, 32, 16), 'learning_rate': 0.023759031518218034, 'batch_size': 128, 'alpha': 1.6815846327195438},
          {'hidden_layers': (128,), 'learning_rate': 0.0015240462867047596, 'batch_size': 32, 'alpha': 2.3313638464100297},
          {'hidden_layers': (64, 64, 64), 'learning_rate': 0.0003148446761691601, 'batch_size': 32, 'alpha': 3.5389208650363404},
          {'hidden_layers': (64, 64, 256, 16), 'learning_rate': 0.0004008219815778804, 'batch_size': 32, 'alpha': 6.802560840943547},
          {'hidden_layers': (128, 64), 'learning_rate': 0.02859818095592624, 'batch_size': 128, 'alpha': 8.739220496981906}]
grad_pen = [0, 0.1, 0.5, 1, 5, 10]

#Datasets
dataset_train = torch.load('./Data/regulation_simulation_data.pt')
dataset_real = torch.load('./Data/regulation_real_data.pt')


#%%
#K-S computation loop
ks_results = np.empty((len(grad_pen), len(config)))
for i, hyper in enumerate(config):
    for j, penalty in enumerate(grad_pen):
        #Train
        data = DataModule(dataset_train, hyper)
        model = LitAutoEncoder(hyper, penalty, n_inputs=dataset_train[:][0].size()[1], n_outputs=1, normalize=normalize)
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
        trainer.fit(model, data)

        #Predict
        y_pred, z_pred = model(dataset_real[:][0])
        dataset_pred = torch.utils.data.TensorDataset(dataset_real[:][0], y_pred, z_pred)

        #Profit & Loss
        real_PL = dataset_real[1:][1] - dataset_real[0][1]
        pred_PL = dataset_pred[1:][1] - dataset_pred[0][1]
        
        #Save results
        ks_results[j, i] = ks_test(real_PL, pred_PL)
        
        
#%%
#Show results
labels = ['Config. ' + str(n+1) for n in range(len(config))]
for i in range(len(config)):
    config[i]['learning_rate'] = round(config[i]['learning_rate'], 5)
    config[i]['alpha'] = round(config[i]['alpha'], 3)

plt.figure(figsize=(14,5))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0.4)

plt.subplot(1,2,1)
plt.plot(grad_pen, ks_results, label=labels)

#Regulation zones
plt.axhline(0.09, linestyle='--', color='orange')
plt.axhline(0.12, linestyle='--', color='red')
plt.axhspan(0, 0.09, facecolor='green', alpha=0.1)
plt.axhspan(0.09, 0.12, facecolor='orange', alpha=0.1)
plt.axhspan(0.12, 0.2, facecolor='red', alpha=0.1)

plt.xlabel('$\lambda$ (gradient penalty)')
plt.ylabel('KS test value')
plt.title('Kolmogorov-Smirnov test')
plt.legend()

plt.subplot(1,2,2)
plt.plot((0,1),(0,1), alpha=0)
plt.axis('off')
for i, label in enumerate(labels):
    j = i - 10 * int(np.floor(i / 10)) #the color cycle only has 10 colors
    plt.text(0, 1-0.1*i, label + ': \n' + str(config[i]), color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j])
    
plt.show()
    