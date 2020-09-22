# -*- coding: utf-8 -*-
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from yolo import YOLO
import cv2
from yolov3.utils import wh2xy


def draw(line):
	"""查看标记的原图"""
	img = cv2.imread(line[0])
	box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
	for m in range(len(box)):
		cnt = wh2xy(box[m]).flatten()
		cnt = np.floor(cnt).astype(np.int)
		cv2.rectangle(img, (cnt[0], cnt[1]), (cnt[2], cnt[3]), (0, 255, 0), 3)
	return img


def see_img():
	yolo = YOLO()
	with open('train2.txt') as f:
		lines = f.readlines()
	np.random.shuffle(lines)
	for k in lines[:50]:
		line = k.split()
		img2 = draw(line)
		try:
			image = cv2.imread(line[0])
			if image is None:
				raise FileNotFoundError('666')
		except Exception:
			print('Open Error! Try again!')
			continue
		else:
			image = yolo.detect_image(image)
			cv2.imshow('img', image)
			cv2.imshow('img1', img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		k = input('233')
		if k.lower() == 'q':
			break
	yolo.close_session()


def see_big_img():
	yolo = YOLO()
	with open('train.txt') as f:
		lines = f.readlines()
	np.random.shuffle(lines)
	for k in lines[:50]:
		line = k.split()
		img2 = draw(line)
		try:
			image = cv2.imread(line[0])
			if image is None:
				raise FileNotFoundError('666')
		except Exception:
			print('Open Error! Try again!')
			continue
		else:
			image = yolo.detect_big_image(image)
			cv2.imshow('img', image)
			cv2.imshow('img1', img2)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		k = input('233')
		if k.lower() == 'q':
			break
	yolo.close_session()


def detect_img():
	yolo = YOLO()
	while True:
		img = input('Input image filename(input Q to exit):')
		if img.lower() == 'q':
			yolo.close_session()
			return
		try:
			image = cv2.imread(img)
			if image is None:
				raise FileNotFoundError('666')
		except Exception:
			print('Open Error! Try again!')
			continue
		else:
			r_image = yolo.detect_big_image(image)
			cv2.imshow('img', r_image)
			cv2.waitKey(0)
			cv2.destroyWindow('img')
	yolo.close_session()


FLAGS = None

if __name__ == '__main__':
	detect_img()
