
############################# Imports ###############################

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from model import G, D
from dataset import Dataset

from IPython import embed

############################# Declarations ##########################

def save_intermediate_results(save_pth, data):

    y_axis = list(np.array(data.view(-1)))
    x_axis = [i for i in range(len(y_axis))]

    plt.clf()
    plt.plot(x_axis, y_axis, '.-', label='line 1', linewidth=2)
    Path('results').mkdir(parents=True, exist_ok=True)
    plt.savefig(save_pth)

def train(config):

    ################################### Model Initalization ####################################

    dataset = Dataset(config['dir_csv'])                                # Class declaraction --> Runs ___init___ with Dataset
    dataloader = DataLoader(dataset, batch_size=1)                      # Built-in class within Pytorch 
     
    '''
    Save Ground Truth Plots
    '''
    if True:

        Path('ground_truths').mkdir(parents=True, exist_ok=True)
        for data in dataloader:
            
            real_data = data['real_data']
            basename = data['basename'][0]

            save_intermediate_results(os.path.join('ground_truths', basename + '.jpg'), real_data)
    
    gen = G().to(config['device'])
    dsc = D().to(config['device'])
    
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=config['lr'])
    optimizer_D = torch.optim.Adam(dsc.parameters(), lr=config['lr'])

    real_label = torch.tensor(1.0).view(1, -1).to(config['device'])     # Tensor of shape (1, 1)
    fake_label = torch.tensor(0.0).view(1, -1).to(config['device'])     # Tensor of shape (1, 1)

    criterion = nn.BCELoss()                                            # Binary Cross Entropy Loss

    fixed_noise = torch.rand((1, 100)).to(config['device'])

    for epoch in range(config['n_epoch']):
    
        for data in dataloader:

            real_data = data['real_data'].to(config['device'])

            ##################### Optimize for Generator ##########################
            
            optimizer_G.zero_grad()
            
            fake_data = gen(fixed_noise)                                # (1, 100) -> (1, 1, 800)
            pred = dsc(fake_data)                                       # (1, 1, 800) -> (1, 1)
            G_loss = criterion(pred, real_label)                        # Train the generator to fool the discriminator
            
            '''
            Optimize
            '''
            G_loss.backward()
            optimizer_G.step()
            
            ##################### Optimize for Discriminator ######################
               
            optimizer_D.zero_grad()

            '''
            Real Input
            '''
            pred = dsc(real_data)                                       # (1, 1, 800) -> (1, 1)
            D_loss_real = criterion(pred, real_label)                   # Train the discriminator to distinguish between real and fake data

            '''
            Fake Input
            '''
            pred = dsc(fake_data.detach())                              # (1, 1, 800) -> (1, 1)
            D_loss_fake = criterion(pred, fake_label)                   # Train the discriminator to distinguish between real and fake data

            '''
            Optimize
            '''
            D_loss_total = (D_loss_real + D_loss_fake) / 2
            D_loss_total.backward()
            optimizer_D.step()
        
        if (((epoch + 1) % config['val_epoch']) == 0):

            Path('results').mkdir(parents=True, exist_ok=True)
            save_intermediate_results(os.path.join('results', 'epoch_%d.jpg'%(epoch + 1)), fake_data.detach().cpu())

        print('[Epoch] %d / %d'%(epoch + 1, config['n_epoch']), end='\r')

if __name__=='__main__':

    '''
    Fixed Seeds for Consistency 
    '''
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    '''
    Configs
    '''
    config = {

        'device' : torch.device('cuda') if (torch.cuda.is_available()) else torch.device('cpu'),    # Device to train with
        'n_epoch' : 400,                                                                           # Number of total epochs to run
        'lr' : 0.0001,                                                                              # Learning Rate
        'dir_csv' : 'real_data',                                                                    # Directory of samples
        'val_epoch' : 20                                                                            # Interval to view results

    }

    '''
    Enter Main Function
    '''
    train(config)


