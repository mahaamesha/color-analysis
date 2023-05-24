import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

project_path = os.getcwd()
input_folder_path = os.path.join(project_path, 'imgs/input/')
output_folder_path = os.path.join(project_path, 'imgs/output/')


def imshow_plt(im):
	plt.imshow( cv.cvtColor(im, cv.COLOR_BGR2RGB) )
	plt.show()
	

# get centroid of every detected cnt, return it as array
def get_centroid(cnts):
	arr = []    # to store centroid(s)
	for cnt in cnts:    # iterate on every object detected
		M = cv.moments(cnt)
		if M['m00'] != 0:
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
		else:   # actually its will be INF, but it causing error message. So, I set it to zero.
			print("Warning: M['m00'] == 0 | returned. arr = %s" %arr)
			return arr

		arr.append( (cx, cy) )  # if cnts > 1, it will be arr = [(centroid_id_1), (centroid_id_2), ...]
	
	return arr

def draw_contour(im_biner, im_ori):
	cnts, hierarchy = cv.findContours(im_biner, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	if len(cnts) > 1:
		cnts = sorted(cnts, key=cv.contourArea, reverse=True)
		cnts = tuple([cnts[0]])
	r = int(1.2/100*im_ori.shape[0])
	im_contour = cv.drawContours(im_ori.copy(), cnts, -1, (0,255,0), r)
	print("Contours:", len(cnts))
	return im_contour, cnts


# draw point for every centroid in centroid_array
def draw_centroid(frame, arr:list):
	# access pixel and change the color
	r = int(1.2/100*frame.shape[0])
	for (xc, yc) in arr:    # remember that, arr = [(cendtroid_id_1), (centroid_id_2), ...]
		cv.circle(frame, center=(xc, yc), radius=r, color=(0,0,255), thickness=-1)
	return frame


def get_roi(frame, centroid:tuple, r:float=20/100):
	mask = np.zeros(frame.shape[:2], np.uint8)	# for background
	r = int(r*frame.shape[0])
	mask = cv.circle(mask, center=centroid, radius=r, color=(255,255,255), thickness=-1)	# draw white circle
	im_roi = cv.bitwise_and(frame, frame, mask=mask)
	return im_roi


def get_im_final(frame, centroid:tuple, r:float=20/100):
	# use frame=im_centroid
	t = int(1.2/100*frame.shape[0])
	r = int(r*frame.shape[0])
	im = cv.circle(frame, center=centroid, radius=r, color=(0,0,255), thickness=t)	# draw the roi circle
	im = cv.circle(im, center=centroid, radius=t, color=(0,0,255), thickness=-1)	# draw the centroid dot
	return im


def im_process(file_path:str):
	im = cv.imread(file_path)
	
	im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
	im_blur = cv.GaussianBlur(im_hsv, (7,7), sigmaX=3)

	lower_hsv, upper_hsv = np.array([0,0,128]), np.array([179,255,255])
	mask = cv.inRange(im_blur, lower_hsv, upper_hsv)     # h<=179; s,v <=255
	
	kernel = np.ones( (7,7), np.uint8 )
	im_opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=20)
	im_closing = cv.morphologyEx(im_opening, cv.MORPH_CLOSE, kernel, iterations=20)

	im_contour, cnts = draw_contour(im_closing, im)

	centroid_arr = get_centroid(cnts)
	im_centroid = draw_centroid(im.copy(), centroid_arr)

	im_roi = get_roi(im.copy(), centroid_arr[0], r=20/100)
	im_final = get_im_final(im_contour.copy(), centroid_arr[0], r=20/100)
	
	plt.subplot(241); plt.title('original'); plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
	plt.subplot(242); plt.title('mask'); plt.imshow(mask, cmap='gray')
	plt.subplot(243); plt.title('opening'); plt.imshow(im_opening, cmap='gray')
	plt.subplot(244); plt.title('closing'); plt.imshow(im_closing, cmap='gray')
	plt.subplot(245); plt.title('contour'); plt.imshow(cv.cvtColor(im_contour, cv.COLOR_BGR2RGB))
	plt.subplot(246); plt.title('centroid'); plt.imshow(cv.cvtColor(im_centroid, cv.COLOR_BGR2RGB))
	plt.subplot(247); plt.title('roi'); plt.imshow(cv.cvtColor(im_roi, cv.COLOR_BGR2RGB))
	plt.subplot(248); plt.title('final'); plt.imshow(cv.cvtColor(im_final, cv.COLOR_BGR2RGB))
	plt.tight_layout()

	fname = os.path.basename(file_path)[:-4]		# without extension
	save_path = os.path.join(output_folder_path, fname)
	plt.savefig(save_path + '_process.jpg', dpi=300)
	cv.imwrite(save_path + '_final.jpg', im_final)
	
	# plt.show()
	return im_roi


# only for single channel
def get_hist(im, mask=None):
	# handle if mask=None
	if not mask: mask = np.zeros(im.shape[:2], np.uint8); mask[:, :] = 255
	masked_im = cv.bitwise_and(im, im, mask=mask)

	fig, ax = plt.subplots(figsize=(8,6))
	hist = cv.calcHist([masked_im], [0], mask, [256], [0,256])
	ax.plot(hist)
	ax.set_xlim(0, 256)
	ax.set_xlabel('intensity'); ax.set_ylabel('frequency')
	plt.show()


# for colored (3 channels) image
def get_rgb_histogram(im, mask=None):
	# handle if mask=None
	if not mask: mask = np.zeros(im.shape[:2], np.uint8); mask[:, :] = 255
	masked_im = cv.bitwise_and(im, im, mask=mask)

	color = ('b','g','r')
	fig, ax = plt.subplots(figsize=(8,6))
	for i,clr in enumerate(color):
		hist = cv.calcHist([masked_im], [i], mask, [256], [0,256])
		ax.plot(hist, color=clr)
	ax.set_xlim(0, 256)
	ax.set_xlabel('intensity'); ax.set_ylabel('frequency')
	plt.show()