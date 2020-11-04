import os
from os import listdir
import cv2 
import numpy as np


path = 'D:/realData' # 路徑自己改
files = listdir(path+'/tmp')
for file in files:
    print(file)
    cap = cv2.VideoCapture(path+'/tmp/'+file)
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoName = file

    currentFrame = 0
    if not os.path.isdir(path+'/tmp/'+ videoName):
        os.mkdir(path+'/tmpFrame/'+ videoName)
        print("----------start "+videoName+'----------')
        while(currentFrame < cap.get(cv2.CAP_PROP_FRAME_COUNT)):  # get frame 
            # path = 'D:/sl_data/train_data/'+ videoName
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            ret, frame = cap.read()   # Capture frame-by-frame 
            # Saves image of the current frame in jpg file
            name = path+'/tmpFrame/'+ videoName +'/frame' + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
            # To stop duplicate images
            currentFrame += 1
    else:
        print("------exist-------")
    print("----------end "+videoName+'----------')
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

