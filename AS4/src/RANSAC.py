'''
RANSAC
-------------------------
Usage:
    RANSAC.py filename configname
    filrname: A points correspondence file (3D-2D)
    configname: RANSAC parameters(prob, nmin, nmax, kmax)
    (e.g. .\RANSAC.py ..\data\correspondingPoints_test_noise0.
txt .\RANSAC.config)
-------------------------
'''
import cv2
import numpy as np
import sys
import random
import math

def main():
    print(__doc__)
    op, ip = readData()
    prob, nmin,nmax, kmax = config()
    inlinerNum, bestM= ransac(op, ip, prob, nmin, nmax, kmax)
    # print(inlinerNum, bestM)
    computeParams(bestM)
    # a1 = matirxA(op,ip)
    # m1 = matrixM(a1)
    # computeParams(m1)

def computeParams(M):
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    b = []
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))
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

def ransac(op, ip, prob, nmin, nmax, kmax):
    w = 0.5
    # k = math.log(1 - prob) / math.log(1 - (w ** nmin))
    k = kmax
    np.random.seed(0)
    count = 0
    inlinerNum = 0
    bestM = None
    a1 = matirxA(op, ip)
    m1 = matrixM(a1)
    fullD = distance(m1, op, ip)
    medianDistance = np.median(fullD)
    t = 1.5 * medianDistance
    n = random.randint(nmin, nmax)
    while(count < k and count < kmax):
        
        index = np.random.choice(len(op), n)
        ranOp, ranIp = np.array(op)[index], np.array(ip)[index]
        A = matirxA(ranOp, ranIp)
        M = matrixM(A)
        d = distance(M, op, ip)
        inliner = []
        for i, d in enumerate(d):
            if d < t:
                inliner.append(i)
        if len(inliner) >= inlinerNum:
            inlinerNum = len(inliner)
            inlinerOp, inlinerIp = np.array(op)[inliner], np.array(ip)[inliner]
            A = matirxA(ranOp, ranIp)
            bestM = matrixM(A)
        if not (w == 0 ):
            w = float(len(inliner))/float(len(ip))
            k = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** n)))
        count += 1;
    return inlinerNum, bestM

def distance(M, op, ip):
    m1 = M[0][:4]
    m2 = M[1][:4]
    m3 = M[2][:4]
    d = []
    for i, j in zip(op, ip):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        # pi = np.concatenate([pi, [1]])
        pi = np.append(pi, 1)
        exi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2))
        d.append(di)
    return d

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
    return M

def config():
    configname = sys.argv[2]
    with open(configname, 'r') as conf:
        prob = float(conf.readline().split()[0])
        kmax = int(conf.readline().split()[0])
        nmin = int(conf.readline().split()[0])
        nmax = int(conf.readline().split()[0])
    return prob, nmin, nmax, kmax

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