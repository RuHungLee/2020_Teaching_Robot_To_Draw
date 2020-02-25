import torch
import os
import cv2
import numpy as np
import dataloader
import time as t
from network import Global
from torchvision.transforms import ToPILImage
pretrained = './pretrained/5_run.pth.tar'
device = torch.device('cuda:0')
model = Global()
model = model.to(device)
model.load_state_dict(torch.load(pretrained))
predict_set = dataloader.Loader(path='/home/eric123/Drawing/test_dataset')
dataloader = torch.utils.data.DataLoader(predict_set , batch_size = 1 , shuffle = False , num_workers = 1)
for i , (x , _) in enumerate(dataloader):
    x = x.cuda()
    out = model(x)
    img = out.reshape(1,109,109).cpu().detach().squeeze(0)
    img = np.array(img)
    print(img.shape)
    os.mkdir(f'./out_test/{i}_run')
    cv2.imwrite(f'./out_test/{i}_run/predict.jpg' , img)
    #predict = trans(img)
    #os.mkdir(f'./output/{i}_Pair')l
    #predict.save(f'./output/{i}_Pair/predict.jpg')


    
    
    '''
    print('------------------------')
    print('predict:',predict)
    target = torch.max(y.view(y.shape[0] , -1) , 1)[1]
    print('target:',target)
    '''
