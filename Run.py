import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import os
import gc
from torch.utils.data import DataLoader
from Arch import Net
import torchvision
import dataload
import gc 

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train(n_epochs, optimizer, model, loss_fn, train_loader, save_file, loss_file):
    
    l_train=[]
    l_val=[]
    
    print('####START####',datetime.datetime.now())

    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        acc_train = 0.0
        counter=0
        
        for sig, lab in train_loader:

            sig = sig.to(device=device); lab = lab.float();
            lab = lab.to(device=device) 

            ops = model(sig)
                       
            loss = loss_fn(ops,lab.unsqueeze(1))
            acc = binary_acc(ops,lab.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            acc_train += acc.item()

            counter +=1

        if epoch%5==0:
            sf2 = save_file.split('.')[0]
            sf2 = sf2+'_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), sf2)

        l_train.append(loss_train/len(train_loader))
        tempa = np.asarray(l_train)
        np.savetxt(loss_file,tempa)
        del tempa

        print('{} Epoch {} | Training loss {} | Accuracy {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader), acc_train/len(train_loader)))
        torch.save(model.state_dict(), save_file)


os.makedirs('Simple',exist_ok=True)
save_file = 'Simple/Arch1.pth'
loss_file = 'Simple/Arch1.txt'

loss_fn = nn.BCEWithLogitsLoss()

batch = 16

epochs = 200

device = 'cuda'
model = Net()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = dataload.SigDataset("train1.csv",transform=torchvision.transforms.Compose([dataload.SigPathToTensor()]))
train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True, drop_last=True)

train(n_epochs=epochs,optimizer=optimizer, model=model,loss_fn=loss_fn, train_loader=train_loader, save_file=save_file, loss_file=loss_file)

gc.collect() 
torch.cuda.empty_cache()
