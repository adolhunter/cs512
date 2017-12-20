'''
Optical Flow Estimation
USAGE: 
    AS6.py [<video_source>]
    e.g. .\AS6.py ..\data\Walk2.mpg
Keys:
    p ..... pause/release current image
    ESC ... exit <<press q twice to exit when video is playing>>
'''
from scipy import ndimage
import numpy as np
import cv2
import sys

HSKernel =np.array([[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]],float)

def main():
    print(__doc__)

    cap = cv2.VideoCapture(sys.argv[1])
    ret, prev = cap.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.GaussianBlur(prevgray,(9,9),2)
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            ch = cv2.waitKey(30)
            if ch == 27:
                break
            if ch == ord('p') or ch == ord('P'):
                print("press 'p' to pause/release current image")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray,(9,9),2)
                # flow = 2 * cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = 2 * HS(prevgray, gray, 0.001, 8)
                prevgray = gray
                cv2.imshow('flow', draw_flow(gray, flow))
                while True:
                    ch2 = cv2.waitKey(1)
                    if ch2 == (ord('p') or ch == ord('P')):
                        cv2.destroyAllWindows()
                        break
                    elif ch2 == 27:
                        break
            cv2.imshow('video', img)
    cap.release()
    cv2.destroyAllWindows()

def computeDerivatives(prev,curr):
    fx = cv2.Sobel(prev,cv2.CV_32F,1,0,ksize=1)
    fy = cv2.Sobel(prev,cv2.CV_32F,0,1,ksize=1)
    ft = curr - prev
    return fx, fy, ft

def HS(im1, im2, alpha, iter):
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])
    U = uInitial
    V = vInitial
    [fx, fy, ft] = computeDerivatives(im1, im2)
    count = 0
    for count in range(iter):
        uAvg = ndimage.convolve(U, HSKernel, mode = 'constant', cval = 0.0)
        vAvg = ndimage.convolve(V, HSKernel, mode = 'constant', cval = 0.0)
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        U = uAvg - fx * der
        V = vAvg - fy * der
        count += 1
    flow = np.dstack((U,V))
    return flow

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

if __name__ == '__main__':
    main()