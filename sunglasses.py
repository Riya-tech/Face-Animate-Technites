import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image
import cvzone

mpDraw=mp.solutions.mediapipe.python.solutions.drawing_utils
mpFace=mp.solutions.mediapipe.python.solutions.face_mesh
face=mpFace.FaceMesh()
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

img=cv2.imread("f1.jpg")
eye1x,eye2x,eye1y,eye2y,x1,y1=0,0,0,0,0,0
imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
result=face.process(imgRGB)
if(result.multi_face_landmarks):
    for faceLMS in result.multi_face_landmarks:
        for id,lm in enumerate(faceLMS.landmark):
            ih,iw,ic=img.shape
            x,y=int(lm.x*iw),int(lm.y*ih)
            if(id==130):
                print(id,x,y)
                eye1x=x
                eye1y=y
            elif(id==359):
                print(id,x,y)
                eye2x=x
                eye2y=y
            elif(id==25):
                x2=x
                # y1=y;
            elif(id==127):
                x1=x;
            elif(id==68):
                y1=y;

                
glass=cv2.imread("glass1.png",cv2.IMREAD_UNCHANGED)

x, y = int(x1)-int((x2-x1)/1.4), int(y1) 
width=int((eye2x-eye1x)*1.8)
ori_wid=glass.shape[1]
scale=width/ori_wid
height=int(scale*glass.shape[0])
glass=cv2.resize(glass,(width,height),interpolation = cv2.INTER_AREA)

imgResult=cvzone.overlayPNG(img,glass,[x,y])



cv2.circle(img,(x,y),2,(255,0,0),2)                    
cv2.imshow("IMAGE",imgResult)
cv2.waitKey(0)
cv2.destroyWindow("IMAGE")
cv2.destroyWindow("Glass")
# 130,359
