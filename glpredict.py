import torch
import numpy as np
import os
import time
import cv2
from PIL import Image
from torchvision.transforms import ToTensor , ToPILImage , ToPILImage
from lmodel import Local
from gmodel import Global

Lpretrained = './pretrained/2_run.pth.tar'
Lmodel = Local()
Lmodel.load_state_dict(torch.load(Lpretrained , map_location=torch.device('cpu')))

Gpretrained = './pretrained/5_run.pth.tar'
Gmodel = Global()
Gmodel.load_state_dict(torch.load(Gpretrained , map_location=torch.device('cpu')))

path = './test/test1.jpg'

#np.set_printoptions(threshold = np.inf)
def dfs(seen , img , img_connected , x , y):
    for i in [-2 , -1 , 0 , 1 , 2]:
        for j in [-2 , -1 , 0 , 1 , 2]:
            if (i,j)!=(0,0):
                nx = x + i
                ny = y + j
                if(not (nx>0 and nx<109 and ny>0 and ny<109)):
                    continue
                if(img[nx][ny]!=0 and (nx,ny) not in seen):
                    seen.append((nx , ny))
                    dfs(seen , img , img_connected , nx , ny)
                    img_connected[nx,ny] = 1

if __name__ == '__main__':

    to_tensor = ToTensor()
    to_pil = ToPILImage()
    imgin = np.array(Image.open(path))
    imgin = np.where(imgin<100 , 0 , 255)
    img = to_tensor(Image.open(path)).float()
    v = to_tensor(np.zeros((109 , 109 , 1))).float()
    le = to_tensor(np.zeros((109 , 109 ,1))).float()
    ls = to_tensor(np.zeros((109 , 109 , 1))).float()
    uv = img
    id = 0

    #total point 
    cnt = 0
    for i in range(109):
        for j in range(109):
            if(imgin[i][j]==255):
                cnt += 1
                
    while(1):

        #break situation
        p = 0
        for i in range(109):
            for j in range(109):
                if v[0,i,j]>0.5:
                    p += 1
        if(cnt-p<5):
            break
        print('total' , cnt)
        print('current' , p)

        gx = torch.cat((v , uv , le , ls) , 0)
        gx = torch.unsqueeze(gx , 0)

        #Global model
        gx = gx
        locate = Gmodel(gx)
        locate = torch.max(locate.view(locate.shape[0] , -1) , 1)[1]
        locate_x = locate/109
        locate_y = locate%109

        #Local model 
        v[0 , locate_x , locate_y ] = 1
        uv[0 , locate_x , locate_y] = 0
        seen = []
        img_c = torch.zeros((109 , 109))
        connected = dfs(seen , imgin , img_c , locate_x , locate_y)
        img_c = torch.unsqueeze(img_c ,0).float()
        head = (locate_x , locate_y)
        x = torch.cat((v , uv , img_c) , 0)
        image = to_pil(img_c)
        x = torch.unsqueeze(x , 0)
        touched = 0

        #clear ls and le 
        le = to_tensor(np.zeros((109 , 109 ,1))).float()
        ls = to_tensor(np.zeros((109 , 109 , 1))).float()

        while(touched < 0.5):
            touched , shifted = Lmodel(x , head)
            shifted = torch.max(shifted , 1)[1]
            shifted_x = (shifted)/5-2
            shifted_y = (shifted)%5-2
            print(touched) 
            #avoid infinity loop
            if shifted_x==0 and shifted_y==0:
                break
            nx = head[0]+shifted_x
            ny = head[1]+shifted_y
            v[0,nx,ny] = 1
            uv[0,nx,ny] = 0
            head = (nx , ny)
            ls[0,nx,ny] = 1
            x = torch.cat((v , uv , img_c) , 0)
            x = torch.unsqueeze(x , 0)
            #print(touched)
        
        le[0,nx,ny] = 1
        image = to_pil(v.cpu().clone())
        image.save(f'./output/t_{id}.jpg')
        id += 1
        first = False
