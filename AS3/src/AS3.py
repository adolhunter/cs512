import cv2
import numpy as np
import sys


def main():
	combine, image1, image2 = getImage()
	print("input key to Process image(press 'H' for help, press 'q' to quit):")
	k = input()
	while k != 'q':
		if k == 'h':
			n = input("the varience of Gussian scale(n):")
			windowSize = input("windowSize :")
			k = input("the weight of the trace in the harris conner detector(k)[0, 0.5]:")
			threshold = input("threshold:")
			print("processing...")
			rst = harris(combine, n, windowSize, k, threshold)
			showWin(rst)
		if k == 'f':
			rst = featureVector(image1, image2)
			showWin(rst)
		if k == 'b':
			rst = betterLocalization(combine)
			showWin(rst)
		if k == 'H':
			help()
		print("input key to Process image(press 'H' for help, press 'q' to quit):")
		k = input()


def getImage():
	if len(sys.argv) == 3:
		image1 = cv2.imread(sys.argv[1])
		image2 = cv2.imread(sys.argv[2])
	else:
			cap = cv2.VideoCapture(0)
			for i in range(0,15):
				retval1,image1 = cap.read()
				retval2,image2 = cap.read()
			if retval1 and retval2:
				cv2.imwrite("capture1.jpg", image1)
				cv2.imwrite("capture2.jpg", image2)
	combine = np.concatenate((image1, image2), axis=1)
	return combine, image1, image2;


def showWin(image):
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def cvt2Gray(image):
	image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image_bw


def smooth(image, n):
	kernel = np.ones((n, n), np.float32)/(n * n)
	dst = cv2.filter2D(image, -1, kernel)
	return dst


def harris(image, n, windowSize, k, threshold):
	n = int(n)
	windowSize = int(windowSize)
	k = float(k)
	threshold = int(threshold)
	copy = image.copy()
	rList = []
	height = image.shape[0]
	width = image.shape[1]	
	offset = int(windowSize / 2)
	image = cvt2Gray(image)
	image = np.float32(image)
	image = smooth(image, n)
	dy, dx = np.gradient(image)
	Ixx = dx ** 2
	Ixy = dy * dx
	Iyy = dy ** 2

	for y in range(offset, height - offset):
			for x in range(offset, width - offset):
				windowIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				windowIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
				Sxx = windowIxx.sum()
				Sxy = windowIxy.sum()
				Syy = windowIyy.sum()
				det = (Sxx * Syy) - (Sxy ** 2)
				trace = Sxx + Syy
				r = det - k *(trace ** 2)
				rList.append([x, y, r])
				if r > threshold:
							copy.itemset((y, x, 0), 0)
							copy.itemset((y, x, 1), 0)
							copy.itemset((y, x, 2), 255)
							cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	return copy
	# def threshBar(threshold):
	# 	newImage = copy.copy()
	# 	for x, y, r in rList:
	# 		if r > threshold * 1000:
	# 					newImage.itemset((y, x, 0), 0)
	# 					newImage.itemset((y, x, 1), 0)
	# 					newImage.itemset((y, x, 2), 255)
	# 					cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
	# 	cv2.imshow("Display window", newImage)

	# def smoothBar(n):
	# 	kernel = np.ones((n, n), np.float32)/(n * n)
	# 	dst = cv2.filter2D(copy, -1, kernel)
	# 	cv2.imshow("Display window", dst)

	# cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	# # cv2.createTrackbar("smoothing", "Display window", 0, 255, smoothBar)
	# # cv2.createTrackbar("Threshold", "Display window", 0, 1000, threshBar)
	# cv2.imshow("Display window", copy)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


def featureVector(image1, image2):
	# Initiate SIFT detector
	orb = cv2.ORB_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(image1,None) # returns keypoints and descriptors
	kp2, des2 = orb.detectAndCompute(image2,None)
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# Match descriptors.
	matches = bf.match(des1,des2)
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	kp1List = []
	kp2List = []
	for m in matches:
		(x1, y1) = kp1[m.queryIdx].pt
		(x2, y2) = kp2[m.trainIdx].pt
		kp1List.append((x1, y1))
		kp2List.append((x2, y2))
	for i in range(0, 50):
		point1 = kp1List[i]
		point2 = kp2List[i]
		cv2.putText(image1, str(i), (int(point1[0]), int(point1[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
		cv2.putText(image2, str(i), (int(point2[0]), int(point2[1])),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
	rst = np.concatenate((image1, image2), axis=1)
	return rst


def betterLocalization(image):
	gray = cvt2Gray(image)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
	dst = np.uint8(dst)

	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

	rst = np.hstack((centroids,corners))
	rst = np.int0(rst)
	image[rst[:,1],rst[:,0]]=[0,0,255]
	image[rst[:,3],rst[:,2]] = [0,255,0]
	return image


def help():
	print("'h': Estimate image gradients and apply Harris corner detection algorithm.")
	print("'b': Obtain a better localization of each corner.")
	print("'f': Compute a feature vector for each corner were detected.\n")


if __name__ == '__main__':
	main()