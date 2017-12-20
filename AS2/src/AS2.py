import cv2
import numpy as np
from scipy import ndimage
import sys


def getImage():
	if len(sys.argv) == 2:
		filename = sys.argv[1]
		image = cv2.imread(filename)
	elif len(sys.argv) < 2:
		cap = cv2.VideoCapture(0)
		for i in range(0,15):
			retval,image = cap.read()
		if retval:
			cv2.imwrite("capture.jpg", image)
	print(image.shape)
	return image

def reload(img):
		return img


def showWin(image):
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def save(image):
	cv2.imwrite("out.jpg", image)



def cvt2Gray1(image):
	image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# showWin(image_bw)
	return image_bw


def cvt2Gray2(image):
	 
	image_bw2 = np.dot(image[...,:3], [0.299, 0.587, 0.114])
	return image_bw2

def cycleColor(image):
	for i in range(0, image.shape[0], 1):
		for j in range(0, image.shape[1], 1):
			k = input()
			if k == 'q':
				return
			else:
				print(image[i][j][0], image[i][j][1],image[i][j][2])


def smooth(image):
	n = 5
	kernel = np.ones((n, n), np.float32)/(n * n)
	dst = cv2.filter2D(image, -1, kernel)
	return dst


def smooth1(image):
	def sliderHandler(n):
		kernel = np.ones((n, n), np.float32)/(n * n)
		dst = cv2.filter2D(image, -1, kernel)
		cv2.imshow("Display window", dst)
	image = cvt2Gray1(image)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("smoothing", "Display window", 0, 255, sliderHandler)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def smooth2(image):
	def sliderHandler(n):
		dst = ndimage.gaussian_filter(image, n, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
		cv2.imshow("Display window", dst)
	cvt2Gray2(image)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("smoothing", "Display window", 0, 10, sliderHandler)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def downSample1(image):
	ds = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
	showWin(image)
	return image


def downSample2(image):
	smooth(image)
	ds = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
	showWin(image)
	return image


def xdrv(image):
	image = cvt2Gray1(image)
	image = smooth(image)
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 5)
	cv2.normalize(sobelx, sobelx, 0, 255, cv2.NORM_MINMAX)
	return sobelx


def ydrv(image):
	image = cvt2Gray1(image)
	image = smooth(image)
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 5)
	cv2.normalize(sobely, sobely, 0, 255, cv2.NORM_MINMAX)
	return sobely


def xydrv(image):
	image = smooth(image)
	sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize = 5)
	return sobel


def gradientVector(image):
	image = cvt2Gray1(image)


def rotation(image):
	def sliderHandler(degree):
		cols = image.shape[0]
		rows = image.shape[1]
		M = cv2.getRotationMatrix2D((rows / 2, cols / 2), degree, 1)
		dst = cv2.warpAffine(image, M, (rows, cols))
		cv2.imshow("Display window", dst)
	image = cvt2Gray1(image)
	cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
	cv2.createTrackbar("smoothing", "Display window", 0, 360, sliderHandler)
	cv2.imshow("Display window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def help():
	print("'i': reload the original image.")
	print("'w': save the current image in to 'out.jpg'.")
	print("'g': convert the image to grayscale using the openCV conversion function.")
	print("'G': conver the image to grayscale using my implementation.")
	print("'c': cycle through the color channels(press 'q' to break).")
	print("'s': convert the image to grayscale and smooth using the openCV function.")
	print("'S': convert the image to grayscale and smooth using my function.")
	print("'d': downsample the image by a factor of 2 without smoothing.")
	print("'D': downsample the image by a factor of 2 with smoothing.")
	print("'x': convert the image to grayscale and perform  convolution with a x derivative filter.")
	print("'y': convert the image to grayscale and perform  convolution with a y derivative filter.")
	print("'m': show the magnitude of the gradient normalized to the range [0, 255].")
	print("'p':convert the image to grayscale and plot the gradient vector of the image every N pixels and let the plotted gradient vector have a length of K.")
	print("'r': convert the image to grayscale and rotate it.\n")
	print("input key to Process image(press 'q' to quit):")

def main():
	image = getImage()
	img = image.copy()
	k = input()
	while k != 'q':
		print("input key to Process image(press 'q' to quit):")
		if k == 'i':
			image = reload(img)
		elif k == 'w':
			save(image)
		elif k == 'g':
			image = cvt2Gray1(image)
		elif k == 'G':
			image = cvt2Gray2(image)
		elif k == 'c':
			cycleColor(image)
		elif k == 's':
			image = smooth1(image)
		elif k == 'S':
			image = smooth2(image)
		elif k == 'd':
			image = downSample1(image)
		elif k == 'D':
			image = downSample2(image)
		elif k == 'x':
			image = xdrv(image)
		elif k == 'y':
			image = ydrv(image)
		elif k == 'm':
			image = xydrv(image)
		elif k == 'p':
			image = gradientVector(image)
		elif k == 'r':
			image = rotation(image)
		elif k == 'h':
			help()
		k = input()

	



if __name__ == '__main__':
	main()