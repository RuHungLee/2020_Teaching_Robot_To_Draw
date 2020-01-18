import numpy as np
import pandas as pd
import cv2
import os

if __name__ == '__main__':
    path = r'/home/eric123/Drawing/output'
    savePath = r'/home/eric123/Drawing/dataset'
    wordList = os.listdir(path)
    wordList.sort()
    for name in wordList:

        filename = os.path.join(path , name)
        word = pd.read_csv(filename , header = None)
        x = word[1]
        y = word[0]
        s = word[2]
        img = np.zeros((109,109))
        lastEndpoint = np.zeros((109,109))
        visited = np.zeros((109,109))
        lastFinished = np.zeros((109,109))
        unvisited = np.zeros((109,109))
        groundtruth = np.zeros((109,109))
        #create picture
        for i in range(0,len(s)):
            img[int(x[i])][int(y[i])] = 1
        #initialize id
        id = 1
        last = -1
        current = s[0]
        for i in range(0,len(s)):
            
            if(i==0):
                #save image
                sp = os.path.join(savePath , f'{name[:-8]}_{id}_run')
                os.mkdir(sp)
                groundtruth[int(x[i])][int(y[i])] = 1
                cv2.imwrite(os.path.join(sp,f'visited.jpg') , visited)
                cv2.imwrite(os.path.join(sp,f'nvisited.jpg') , img*255)
                cv2.imwrite(os.path.join(sp,f'lastEndpoint.jpg') , lastEndpoint)
                cv2.imwrite(os.path.join(sp,f'lastFinished.jpg') , lastFinished)
                cv2.imwrite(os.path.join(sp,f'label.jpg') , groundtruth*255)
                groundtruth.fill(0)
                id+=1
            elif(i!=len(s)-1):
                last = current
                current = s[i]
                if(last==current):
                    visited[int(x[i])][int(y[i])] = 1
                    lastFinished[int(x[i])][int(y[i])] = 1
                else:
                    # print(savePath)
                    sp = os.path.join(savePath , f'{name[:-8]}_{id}_run')
                    os.mkdir(sp)
                    lastEndpoint[int(x[i-1])][int(y[i-1])] = 1
                    groundtruth[int(x[i])][int(y[i])] = 1
                    unvisited = img-visited
                    #save image
                    cv2.imwrite(os.path.join(sp,f'visited.jpg') , visited*255)
                    cv2.imwrite(os.path.join(sp,f'nvisited.jpg') , unvisited*255)
                    cv2.imwrite(os.path.join(sp,f'lastEndpoint.jpg') , lastEndpoint*255)
                    cv2.imwrite(os.path.join(sp,f'lastFinished.jpg') , lastFinished*255)
                    cv2.imwrite(os.path.join(sp,f'label.jpg') , groundtruth*255)
                    id+=1
                    #clear 
                    lastEndpoint.fill(0)
                    lastFinished.fill(0)
                    groundtruth.fill(0)
