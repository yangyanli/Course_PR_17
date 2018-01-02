# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:53:06 2017

@author: Hanlin
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json
import cv2  

def get_model():
    # Load the model architecture
    model = model_from_json(open('model_architecture.json').read())
    # Load the model weights
    model.load_weights('model_weights.h5')
    return model

if __name__=='__main__':
    model = get_model()
    img = cv2.imread("D:/20171128115348.jpg") # open colour image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray, 124, 255, cv2.THRESH_BINARY_INV)  
    _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    rects = [cv2.boundingRect(contours) for contours in contours]
    i=0
    digit=[]
    rect1=[]
    rect2=[]
    for rect in rects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 1) 
            leng = int(rect[3] * 1.6)
            
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            length=(leng-rect[3])//2
            weight=(leng-rect[2])//2
            print(length,weight)
            if (i/3==0):
                cut=binary[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                cv2.imshow("roi %s" % (i),cut)
            if (length>0 & weight>0):
                border=cv2.copyMakeBorder(cut, top=length, bottom=length, left=weight, right=weight, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
            roi = binary[pt1:pt1 + leng, pt2:pt2 + leng]
            if (roi.size>0):
                rect1.append(rect[0])
                rect2.append(rect[1])
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                roi = roi/255
                i=i+1
                roi=np.asarray(roi,dtype='float32')
                digit.append(roi)
    digit=np.array(digit)
    digit=digit.reshape(-1,1,28,28)           
    y_test = np.argmax(model.predict(digit,batch_size=50,verbose=1),axis=1)
    for i in range(len(y_test)):   
        cv2.putText(img, str(y_test[i]), (rect1[i], rect2[i]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    
    cv2.imshow("img", img) 
    cv2.waitKey(0) 


