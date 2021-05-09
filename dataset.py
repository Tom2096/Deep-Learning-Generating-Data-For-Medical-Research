
############################# Imports ###############################

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os
import torch
import random
import numpy as np
import pandas as pd

from IPython import embed

############################# Declarations ##########################

class Dataset:

    def __init__(self, dir_csv, shuffle=False):
        
        self.list_dir = os.listdir(dir_csv)                                             # List of all files in the given directory

        self.list_dir = [os.path.join(dir_csv, filename) for filename in self.list_dir] # Merge root directory with filenames for later usage

        if (shuffle):
            random.shuffle(self.list_dir)

    def __getitem__(self, idx):

        pth_csv = self.list_dir[idx]
        basename = os.path.splitext(os.path.basename(pth_csv))[0]

        df = pd.read_csv(pth_csv)

        MPL = list(df['MPL'].dropna())                          # Read column 'MPL' and drop all empty rows
        MPL = [v for v in MPL if (isinstance(v, float))]        # Filter list so that it only contains float values
        MPL = torch.tensor(MPL)                                 # Convert list to tensor for training

        min_sample = min(MPL)
        max_sample = max(MPL)
        span = max_sample - min_sample

        normalized = (MPL - min_sample) / span                  # Normalize
        real_data = torch.zeros(800, dtype=torch.float32)
        real_data[:normalized.shape[0]] = normalized            # Pad Shape to (800,)
        real_data = real_data.view(1, -1)                        # Reshape to (1, 800)

        return {
            'real_data' : real_data,
            'basename' : basename
        }
    
    def __len__(self):

        return len(self.list_dir)