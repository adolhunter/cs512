'''
extract feature points
use the openCV functions
-------------------------
Usage:
    ExtractFeats.py filename
    filename : an image contain 3d chessboard
    (E.g. .\ExtractFeats.py ..\data\chessboard.jpg)
-------------------------
Keys:
    select image window
    press any key to exit
-------------------------   
Output:
    correspondencePoints.txt
    A point correspondence file (3D-2D)
'''
import cv2
import numpy as np
import sys

def main():
    print(__doc__)
    image = getImage()
    extractFeats(image)


def getImage():
    if len(sys.argv) == 2:
        image = cv2.imread(sys.argv[1])
    else:
        print("please, input one image.")
    return image;

def extractFeats(image):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret:
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (7,6), corners2, ret)
        cv2.imshow("image", image)

        file = open("correspondencePoints.txt", "w")
        for i, j in zip(objp, corners.reshape(-1,2)):
            file.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(j[0]) + ' ' + str(j[1]) + '\n')
        file.close()

        cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()