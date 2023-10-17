import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd 
import pickle
from random import randrange

class SigPathToTensor(object):

    def __init__(self):
        pass

    def __call__(self, path):

        with open(path, 'rb') as f:
            cap = pickle.load(f)
        f.close()

        cap = (cap-cap.min())/(cap.max()-cap.min()) #z-score normalization
        cap = np.expand_dims(cap,-1);

        cap = torch.from_numpy(cap)

        cap = cap.permute(-1, 0)
        cap = cap.float() 

        return cap
        
class SigDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        sig = self.dataframe.iloc[index].path
        lab = self.dataframe.iloc[index].label

        if self.transform: sig = self.transform(sig)

        return sig, lab
