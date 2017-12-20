'''
non-planar Calibration
-------------------------
Usage:
    Calibrate.py filename
    filename : A points correspondence file (3D-2D)
    (E.g. .\Calibrate.py ..\data\correspondingPoints_test.txt)
-------------------------
output:
    - print intrinsic and extrinsic parameters
    - print mean square error
'''
import cv2
import numpy as np
import sys

def main():
    print(__doc__)
    op, ip = readData()
    A = matirxA(op, ip)
    a1, a2, a3, b, M = matrixM(A)
    computeParams(a1, a2, a3, b)
    meanSquareError(M,op,ip)

def computeParams(a1, a2, a3, b):
    np.set_printoptions(formatter={'float': "{0:.6f}".format})
    normP = 1 / np.linalg.norm(a3.T)
    u0 = normP ** 2 * (a1.T.dot(a3))
    v0 = normP ** 2 * (a2.T.dot(a3))
    a22 = a2.T.dot(a2)
    av = np.sqrt(normP ** 2 * a22 - v0 ** 2)
    a1xa3 = np.cross(a1.T, a3.T)
    a2xa3 = np.cross(a2.T, a3.T)
    s = (normP ** 4) / av * a1xa3.dot(a2xa3.T)
    a12 = a1.T.dot(a1)
    au = np.sqrt(normP ** 2 * a12 - s ** 2 - u0 ** 2)
    Kstar = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    littleE = np.sign(b[2])
    Tstar = littleE * normP * np.linalg.inv(Kstar).dot(b).T
    r3 = littleE * normP * a3
    r1 = normP ** 2 / av * a2xa3
    r2 = np.cross(r3, r1)
    Rstar = np.array([r1.T, r2.T, r3.T])
    print("--------------------------------------")
    print("u0, v0 = %f, %f\n" % (u0, v0))
    print("alphaU,alphaV = %f, %f\n" % (au, av))
    print("s = %f\n" % s)
    print("K* = %s\n" % Kstar)
    print("T* = %s\n" % Tstar)
    print("R* = %s\n" % Rstar)

def meanSquareError(M, op, ip):
    m1 = M[0][:4]
    m2 = M[1][:4]
    m3 = M[2][:4]
    mse = 0
    for i, j in zip(op, ip):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        exi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        mse += ((xi - exi) ** 2 + (yi - eyi) ** 2)
    mse = mse / len(op)
    print("--------------------------------------")
    print("Mean Square Error = %s\n" % mse)

def matirxA(op, ip):
    A = []
    zero = np.zeros(4)
    for i, j in zip(op, ip):
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        xipi = j[0] * pi
        yipi = j[1] * pi
        A.append(np.concatenate([pi, zero, -xipi]))
        A.append(np.concatenate([zero, pi, -yipi]))
    # print(np.array(A))
    return np.array(A)

def matrixM(A):
    M = []
    u, s, v = np.linalg.svd(A, full_matrices = True)
    M = v[-1].reshape(3, 4)
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = []
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))
    return a1, a2, a3, b, M

def readData():
    filename = sys.argv[1]
    op,ip = [], []
    with open(filename) as f:
        data = f.readlines()
        for i in data:
            pt = i.split()
            op.append([float(p) for p in pt[:3]])
            ip.append([float(p) for p in pt[3:]])
    return op, ip

if __name__ == '__main__':
    main()