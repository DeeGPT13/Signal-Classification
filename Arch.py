import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

class Net(nn.Module):
    
    def __init__(self):
    
        super(Net, self).__init__()
        
        self.c1a = nn.Conv1d(1, 16, kernel_size = 8, stride=1, padding=3)
        self.c1b = nn.Conv1d(16, 16, kernel_size = 8, stride=1,padding=3) 

        self.c2a = nn.Conv1d(16, 32, kernel_size = 8, stride=1,padding=3)
        self.c2b = nn.Conv1d(32, 32, kernel_size = 8, stride=1,padding=3) 

        self.c3a = nn.Conv1d(32, 64, kernel_size = 8, stride=1,padding=3)
        self.c3b = nn.Conv1d(64, 64, kernel_size = 8, stride=1,padding=3) 
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(8, stride=2)
        self.glob = nn.MaxPool1d(152, stride=1)
    
        self.bn0 = nn.BatchNorm1d(16)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.linear1 = nn.Linear(64,16)
        self.linear2 = nn.Linear(16,1)
        
    def forward(self, x):
                
        x = self.relu(self.bn0(self.c1a(x)))
        x = self.relu(self.bn0(self.c1b(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn1(self.c2a(x)))
        x = self.relu(self.bn1(self.c2b(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.c3a(x)))
        x = self.relu(self.bn2(self.c3b(x)))
        x = self.glob(x)
        
        x = torch.squeeze(x)
        
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x 
