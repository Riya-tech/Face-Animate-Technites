import numpy as np
import cv2

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
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


img = cv2.imread("media/f2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img, 1.2, 5, 0, (120, 120), (350, 350))

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

        specs = cv2.resize(specsOri, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
        cigar = cv2.resize(cigarOri, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)
        mustache = cv2.resize(musOri, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

        transparentOverlay(face_glass_roi_color, specs)
        transparentOverlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))
        transparentOverlay(face_mus_roi_color, mustache)

cv2.imshow('Thug Life', img)
cv2.waitKey(0)
