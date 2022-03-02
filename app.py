import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image
import cvzone
from flask import Flask, escape, request, render_template, Response, make_response, redirect
import matplotlib.pyplot as plt


mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
mpFace = mp.solutions.mediapipe.python.solutions.face_mesh
face = mpFace.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

vid = cv2.VideoCapture(0)


def filter1():

    while True:

        ret, frame = vid.read()

        img = frame

        eye1x, eye2x, eye1y, eye2y, x1, y1 = 0, 0, 0, 0, 0, 0
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face.process(imgRGB)
        if(result.multi_face_landmarks):
            for faceLMS in result.multi_face_landmarks:
                for id, lm in enumerate(faceLMS.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    if(id == 130):
                        # print(id, x, y)
                        eye1x = x
                        eye1y = y
                    elif(id == 359):
                        # print(id, x, y)
                        eye2x = x
                        eye2y = y
                    elif(id == 25):
                        x2 = x
                        # y1=y;
                    elif(id == 127):
                        x1 = x
                    elif(id == 68):
                        y1 = y

        glass = cv2.imread("media/glass1.png", cv2.IMREAD_UNCHANGED)

        x, y = int(x1)-int((x2-x1)/1.4), int(y1)
        width = int((eye2x-eye1x)*1.8)
        ori_wid = glass.shape[1]
        scale = width/ori_wid
        height = int(scale*glass.shape[0])
        glass = cv2.resize(glass, (width, height),
                           interpolation=cv2.INTER_AREA)

        imgResult = cvzone.overlayPNG(img, glass, [x, y])

        cv2.circle(img, (x, y), 2, (255, 0, 0), 2)

        # cv2.imshow("IMAGE", imgResult)

        ret2, buffer2 = cv2.imencode('.jpg', imgResult)
        frame = buffer2.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# filter 2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

specsOri = cv2.imread('media/glass.png', -1)
cigarOri = cv2.imread('media/cigar.png', -1)
musOri = cv2.imread('media/mustache.png', -1)


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + \
                (1 - alpha) * src[x + i][y + j]
    return src


# img = cv2.imread("media/f2.jpg")

def filter2():
    while True:
        ret, img = vid.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            img, 1.2, 5, 0, (120, 120), (350, 350))

        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                glassMin = int(y + 1.5 * h / 5)
                glassMax = int(y + 2.5 * h / 5)
                sh_glass = glassMax - glassMin

                cigarMin = int(y + 4 * h / 6)
                cigarMax = int(y + 5.5 * h / 6)
                sh_cigar = cigarMax - cigarMin

                musMin = int(y + 3.5 * h / 6)
                musMax = int(y + 5 * h / 6)
                sh_mus = musMax - musMin

                face_glass_roi_color = img[glassMin:glassMax, x:x + w]
                face_cigar_roi_color = img[cigarMin:cigarMax, x:x + w]
                face_mus_roi_color = img[musMin:musMax, x:x + w]

                specs = cv2.resize(specsOri, (w, sh_glass),
                                   interpolation=cv2.INTER_CUBIC)
                cigar = cv2.resize(cigarOri, (w, sh_cigar),
                                   interpolation=cv2.INTER_CUBIC)
                mustache = cv2.resize(
                    musOri, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

                transparentOverlay(face_glass_roi_color, specs)
                transparentOverlay(face_cigar_roi_color, cigar,
                                   (int(w / 2), int(sh_cigar / 2)))
                transparentOverlay(face_mus_roi_color, mustache)

        # cv2.imshow('Thug Life', img)
        # cv2.waitKey(0)

        ret2, buffer2 = cv2.imencode('.jpg', img)
        frame = buffer2.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# filter 3


def edge_mark(img,line_size,blur_value):
    # input: Gray Scale Image
    # output: Edges of images
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_blur=cv2.medianBlur(gray_img,blur_value)
    
    edges=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)
    return edges

def color_quantization(img,k):
    
    #Transform imgae
    data=np.float32(img).reshape((-1,3))
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    
    
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    return result



def filter3():
    while True:
        # filename = "media/dhoni.jpg"
        ret, filename = vid.read()
        
        img = filename
        edges=edge_mark(img,line_size=5,blur_value=7)
        img=color_quantization(img,k=15)
        blurred=cv2.bilateralFilter(img,d=7,sigmaColor=200,sigmaSpace=200)
        c=cv2.bitwise_and(blurred,blurred,mask=edges)       
        
        ret2, buffer2 = cv2.imencode('.jpg', c)
        frame = buffer2.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)


@app.route('/cartoon')
def cartoon():
    return render_template('index.html')

@app.route('/ThugLife')
def ThugLife():
    return render_template('thuglife.html')

@app.route('/sunglasses')
def sunglasses():
    return render_template('sunglasses.html')

@app.route('/video1')
def video1():
    return Response(filter3(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video2')
def video2():
    return Response(filter2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video3')
def video3():
    return Response(filter1(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.debug = True
    app.run()
    
    
