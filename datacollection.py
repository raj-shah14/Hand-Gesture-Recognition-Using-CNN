import cv2
import numpy as np
import time
import pyautogui
import os



#Creaing folder for data
file_path="C:/Users/Raj Shah/Downloads/AHD_Project/data/Testdata"
if not os.path.exists(file_path):
    os.makedirs(file_path)

path="C:/Users/Raj Shah/Downloads/AHD_Project/data/Testdata"
#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 700)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 400)

h,s,v = 150,150,150
j=10             #Gesture No.
while(1):
    if(cv2.waitKey(2)==99):
        for i in range(1,501):
        
            ret, frame = cap.read()

            cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
            crop_frame=frame[100:300,100:300]
             #Blur the image
            #blur = cv2.blur(crop_frame,(3,3))
            blur = cv2.GaussianBlur(crop_frame, (3,3), 0)    
                #Convert to HSV color space
            hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
            
            #Create a binary image with where white will be skin colors and rest is black
            mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
            #cv2.imshow('Masked',mask2)

            kernel_square = np.ones((11,11),np.uint8)
            kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

            
            #Perform morphological transformations to filter out the background noise
            #Dilation increase skin color area
##            #Erosion increase skin color area
##            dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
##            erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
##            dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
##            filtered = cv2.medianBlur(dilation2,5)
##            kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
##            dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
##            kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
##            dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
##            median = cv2.medianBlur(dilation2,5)
##            ret,thresh = cv2.threshold(median,127,255,0)
            med=cv2.medianBlur(mask2,5)
            
            
            cv2.imshow('main',frame)
            cv2.imshow('masked',med)
        
            
        #cv2.imshow('masked_',thresh)
            res_size=cv2.resize(med,(50,50))
            cv2.imwrite(os.path.join(path,"gest"+str(j)+"_"+str(i)+".jpg"),res_size)
            cv2.imshow('res',res_size)
            time.sleep(0.05)
            print (i)
         #close the output video by pressing 'ESC'
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

    else:
        ret, frame = cap.read()
        
        cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0)
        crop_frame=frame[100:300,100:300]
         #Blur the image
        #blur = cv2.blur(crop_frame,(3,3))
        blur = cv2.GaussianBlur(crop_frame, (3, 3), 0)    
            #Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
         
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        #cv2.imshow('Masked',mask2)

        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

##        MIN = np.array([0,30,60],np.uint8)
##        MAX = np.array([20,150,179],np.uint8) #HSV: V-79%
##        HSVImg = cv2.cvtColor(crop_frame,cv2.COLOR_BGR2HSV)	
##        filterImg = cv2.inRange(HSVImg,MIN,MAX) #filtering by skin color
##        
##        filterImg = cv2.erode(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))) #eroding the image
##        filterImg = cv2.dilate(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))) #dilating the image
##        filterImg = cv2.erode(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))) #eroding the image
##        filterImg = cv2.dilate(filterImg,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))) #dilating the image
##        filterImg = cv2.medianBlur(filterImg,5)
##        cv2.imshow('test',filterImg)        
      #Perform morphological transformations to filter out the background noise
        #Dilation increase skin color area
        #Erosion increase skin color area
##        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
##        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
##        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
##        filtered = cv2.medianBlur(dilation2,5)
##        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
##        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
##        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
##        dilation3 = cv2.dilate(dilation2,kernel_ellipse,iterations = 1)
##        median = cv2.medianBlur(dilation3,5)
##        ret,thresh = cv2.threshold(median,127,255,0)
        med=cv2.medianBlur(mask2,5)

        
        cv2.imshow('main',frame)
        cv2.imshow('masked',med)
    
        
     #close the output video by pressing 'ESC'
        k = cv2.waitKey(2) & 0xFF
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()
