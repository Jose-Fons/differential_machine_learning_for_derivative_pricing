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

from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from itertools import product
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
# Hyperparameter tuning
epochs = 25
normalize = True
grad_pen = 10

config = {
    'hidden_layers': tune.choice([layer for r in [2,3] for layer in product([16, 32, 64, 128, 256], repeat=r)]),
    # 'hidden_layers': tune.choice([layer for r in [1,3,5] for layer in product([32, 64, 128], repeat=r)]),
    # 'hidden_layers': tune.choice([tuple([nodes] * layers) for nodes in [16, 32, 64, 128, 256] for layers in range(1,6)]),
    # 'hidden_layers': tune.choice([(256, 128, 64), (128, 64, 32), (64, 32, 16)]),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([32, 64, 128]),
    'alpha': tune.uniform(0, 10),
}

def train_tune(config, epochs=25):
    #Tune changes someway the working directory, so this part needs a complete archive access path
    dataset = torch.load('C:/Users/josef/OneDrive - Universitat de Valencia/TFM/Pricer/Data/regulation_simulation_data.pt')
    data = DataModule(dataset, config)
    model = LitAutoEncoder(config, grad_pen, n_inputs=dataset[:][0].size()[1], n_outputs=1, normalize=normalize)
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    trainer.fit(model, data)

scheduler = ASHAScheduler(max_t=epochs, grace_period=1, reduction_factor=2)
reporter = CLIReporter(parameter_columns=['hidden_layers', 'learning_rate', 
                                          'batch_size', 'alpha'], 
                        metric_columns=['loss', 'training_iteration'])

result = tune.run(train_tune, config=config, progress_reporter=reporter, 
                  scheduler=scheduler, num_samples=200, metric='loss', mode='min',
                  raise_on_failed_trial=False)


#%%
# Train with best hyperparameters
epochs = 25
normalize = True
grad_pen = 10
config = {'hidden_layers': (128, 256, 64, 16), 'learning_rate': 0.0020434855190057453, 'batch_size': 64, 'alpha': 1.147389576148048}
# loss=8.61778564435248e-05

dataset = torch.load('./Data/geometric_simulation_data.pt')
data = DataModule(dataset, config)
model = LitAutoEncoder(config, grad_pen, n_inputs=dataset[:][0].size()[1], n_outputs=1, normalize=normalize)
trainer = pl.Trainer(max_epochs=epochs, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
trainer.fit(model, data)


#%%
#Payoff and gradient prediction
dataset = torch.load('./Data/geometric_real_data.pt')
y_pred, z_pred = model(dataset[:][0])
n_assets = int(dataset[:][0].size()[1] / 2)
loss = nn.MSELoss()(y_pred, dataset[:][1]) + grad_pen * torch.mean(torch.pow((z_pred - dataset[:][2]), 2), 0).sum()

#Save results
dataset_pred = torch.utils.data.TensorDataset(dataset[:][0], y_pred, z_pred)
torch.save(dataset_pred, './Data/predicted_data.pt')

#Graphs function
def plot_results(x, y, y_pred, title, xlab, ylab):
    x = x.detach().numpy()
    y = y.detach().numpy()
    y_pred = y_pred.detach().numpy()
    
    plt.plot(x, y, ".", label="Ground Truth")
    plt.plot(x, y_pred, ".r", markersize=3, alpha=0.5, label="Prediction")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()

#Round configuration values
config['learning_rate'] = round(config['learning_rate'], 5)
config['alpha'] = round(config['alpha'], 3)

#Show results
plt.figure(figsize=(18, 5*n_assets))
for i in range(n_assets):
    plt.subplot(n_assets, 3, 3*(i+1)-2)
    plot_results(dataset[:][0][:, i], dataset[:][1], y_pred, 'Price', 'S'+str(i+1), '$V_0$')
    plt.subplot(n_assets, 3, 3*(i+1)-1)
    plot_results(dataset[:][0][:, i], dataset[:][2][:, i], z_pred[:, i], 'Delta', 'S'+str(i+1), '$\Delta$'+str(i+1))
    plt.subplot(n_assets, 3, 3*(i+1))
    plot_results(dataset[:][0][:, i], dataset[:][2][:, n_assets+i], z_pred[:, n_assets+i], 'Vega', 'S'+str(i+1), '$\\nu$'+str(i+1))

_ = plt.suptitle("European call predictions")

#Print hyperparameter values
_ = plt.figtext(0, 0.86, "Data normalization:")
_ = plt.figtext(0, 0.83, str(normalize), color="blue")
_ = plt.figtext(0, 0.76, "Prediction loss:")
_ = plt.figtext(0, 0.73, str(round(loss.item(), 7)), color="blue")
for i, (key, value) in enumerate(config.items()):
    _ = plt.figtext(0, 0.76-0.1*(i+1), key + ":")
    _ = plt.figtext(0, 0.73-0.1*(i+1), str(value), color="blue")
_ = plt.figtext(0, 0.76-0.1*(len(config.items())+1), "$\lambda$ (Gradient penalty):")
_ = plt.figtext(0, 0.73-0.1*(len(config.items())+1), str(grad_pen), color="blue")
plt.show()


#%%
#Best hyperparameters combinations

# config = {'hidden_layers': (64, 256, 32), 'learning_rate': 0.0009976629846684705, 'batch_size': 64, 'alpha': 0.11631924590986942}
# loss=1.2394721125019714e-05

# config = {'hidden_layers': (128, 256, 64, 16), 'learning_rate': 0.0020434855190057453, 'batch_size': 64, 'alpha': 1.147389576148048}
# loss=8.61778564435248e-05

# config = {'hidden_layers': (256, 64, 64), 'learning_rate': 0.003242405561782988, 'batch_size': 64, 'alpha': 0.7843895203261764}
# loss=0.00031712265922966427

# config = {'hidden_layers': (16, 16, 64), 'learning_rate': 0.0012477005179843312, 'batch_size': 128, 'alpha': 1.887009372781434}
# loss=inf

# config = {'hidden_layers': (256, 128, 64, 32), 'learning_rate': 0.026235662270740853, 'batch_size': 64, 'alpha': 1.8200700322792285}
# loss=inf

# config = {'hidden_layers': (256, 128, 64), 'learning_rate': 0.0002614783398977329, 'batch_size': 32, 'alpha': 1.159029241344316}
# loss=inf

# config = {'hidden_layers': (64, 32, 16), 'learning_rate': 0.023759031518218034, 'batch_size': 128, 'alpha': 1.6815846327195438}
# loss=inf

# config = {'hidden_layers': (128,), 'learning_rate': 0.0015240462867047596, 'batch_size': 32, 'alpha': 2.3313638464100297}
# loss=inf

#Maybe

# config = {'hidden_layers': (64, 64, 64), 'learning_rate': 0.0003148446761691601, 'batch_size': 32, 'alpha': 3.5389208650363404}
# loss=inf

# config = {'hidden_layers': (64, 64, 256, 16), 'learning_rate': 0.0004008219815778804, 'batch_size': 32, 'alpha': 6.802560840943547}
# loss=inf

#Never

# {'hidden_layers': (128, 64), 'learning_rate': 0.02859818095592624, 'batch_size': 128, 'alpha': 8.739220496981906}
# loss=inf