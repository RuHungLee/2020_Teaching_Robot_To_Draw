import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset

datapath = '/home/eric123/Drawing/dataset'

def loader(path , mode='Image'):
    to_tensor = ToTensor()
    to_pil = ToPILImage()
    visited = to_tensor(Image.open(os.path.join(path , 'visited.jpg')))
    unvisited =  to_tensor(Image.open(os.path.join(path , 'nvisited.jpg')))
    lastEnd = to_tensor(Image.open(os.path.join(path , 'lastEndpoint.jpg')))
    lastLine = to_tensor(Image.open(os.path.join(path , 'lastFinished.jpg')))
    x = torch.cat((visited,unvisited,lastEnd,lastLine),0)
    y = to_tensor(Image.open(os.path.join(path , 'label.jpg')))
    return x , y

class Loader(Dataset):
    def __init__(self,loader = loader,path = datapath):
        self.path = path
        self.loader = loader
        self.path_list = os.listdir(path)
    def __len__(self):
        return len(self.path_list)
    def __getitem__(self,idx):
        x , y = self.loader(os.path.join(self.path , self.path_list[idx]))
        return x , y
import time as t 
if __name__ == '__main__':
    torch.set_printoptions(edgeitems = 10 , precision=5)
    train_set = Loader()
    dataloader = torch.utils.data.DataLoader(train_set , batch_size = 1 , shuffle = False , num_workers = 1)
    print('dataset num is:',len(dataloader))
    for i , (x , y) in enumerate(dataloader):
        print(f'{i}------------------')
        out1 = y.view(y.shape[0] , -1)
        print(torch.sort(out1)[0])
        t.sleep(2)
        out2 = out1.to(dtype = torch.long)
        print(torch.sort(out2)[0])
        t.sleep(2)
        print(torch.max(y.view(y.shape[0] , -1) , 1)[1])

