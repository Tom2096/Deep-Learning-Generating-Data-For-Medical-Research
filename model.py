
############################# Imports ###############################

import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

############################ Declaractions ##########################

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        
        self.linear = nn.Linear(100, 25600)

        self.convolutions = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),   # (1, 512, 50) -> (1, 256, 100)
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),   # (1, 256, 100) -> (1, 128, 200)
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),    # (1, 128, 200) -> (1, 64, 400)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),      # (1, 64, 400) -> (1, 1, 800)
            nn.Sigmoid()
        )

    def forward(self, input):
        
        X = self.linear(input)          # (1, 100) -> (1, 25600)
        X = X.view(1, 512, 50)          # (1, 25600) -> (1, 512, 50)
        X = F.relu(X)

        X = self.convolutions(X)        # (1, 512, 50) -> (1, 1, 800)
        
        return X                        # (1, 1, 800)

class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1),               # (1, 1, 800) -> (1, 64, 400)
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),             # (1, 64, 400) -> (1, 128, 200)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),            # (1, 128, 200) -> (1, 256, 100)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),            # (1, 256, 100) -> (1, 512, 50)
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        self.linear = nn.Linear(25600, 1)

    def forward(self, input):

        X = self.convolutions(input)    # (1, 1, 800) -> (1, 512, 50)
        X = X.view(1, 25600)            # (1, 512, 50) -> (1, 25600)
        X = self.linear(X)              # (1, 25600) -> (1, 1)
        X = torch.sigmoid(X)            # (1, 1)

        return X                        # (1, 1)