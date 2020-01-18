import torch
import torch.nn as nn
import torch.optim as optim
import loss
import numpy as np
import Resnet
import dataloader
from torch.nn import functional as F
from visdom import Visdom
from network import Global
import time as t

Epoch = 10

if __name__ == '__main__':

    visdom_server = Visdom(port=3387)
    device = torch.device('cuda:0')
    model = Global()
    model = model.to(device)
    train_set = dataloader.Loader(path='/home/eric123/Drawing/dataset')
    val_set = dataloader.Loader(path='/home/eric123/Drawing/val_dataset')
    dataloader = torch.utils.data.DataLoader(train_set , batch_size = 5 , shuffle = True , num_workers = 1)
    val_dataloader = torch.utils.data.DataLoader(val_set , batch_size = 5 , shuffle = False , num_workers = 1)
    crossEntropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    time = 1
    
    print('The number of train dataloader:' , len(dataloader))
    print('The number of val dataloader:' , len(val_dataloader))

    for epoch in range(1 , Epoch):
        
        # ===============TRAINING
        total_loss = 0
        for i , (x , gt) in enumerate(dataloader):
            x = x.to(device)
            gt = torch.where(gt<0.5,torch.full_like(gt,0),torch.full_like(gt,1.001))
            gt = gt.to(device,dtype=torch.long)
            gt = gt.view(gt.shape[0] , -1)
            predict = model(x)
            loss = crossEntropy(predict, torch.max(gt , 1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if(i%1000==0):
                print(f'{epoch}_{i}_loss: {total_loss/(i+1)}')
                visdom_server.line([total_loss/(i+1)],[time],win='p_loss',env='robotarm',update='append')
                time += 1
        print(f'{epoch}epoch_train:{total_loss}')
        total_loss /= len(dataloader)
        visdom_server.line([total_loss],[epoch],win='loss',env='robotarm',update='append')
        torch.save(model.state_dict() , f'./pretrained/{epoch}_run.pth.tar')

        #=================VALIDATION
        total_loss = 0 
        for i , (x , gt) in enumerate(val_dataloader):
            x = x.to(device)
            gt = torch.where(gt<0.5,torch.full_like(gt,0),torch.full_like(gt,1.001))
            gt = gt.to(device,dtype=torch.long)
            gt = gt.view(gt.shape[0] , -1)
            predict = model(x)
            loss = crossEntropy(predict, torch.max(gt , 1)[1])
            total_loss += loss.item()
        print(f'{epoch}epoch_val:{total_loss}')
        total_loss /= len(val_dataloader)
        visdom_server.line([total_loss],[epoch],win='val_loss',env='robotarm',update='append')
            
            



