# -*- coding: utf-8 -*-
"""
 @Time    : 2019/4/5 下午 11:11
 @Author  : Err0r log
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

from yolov3.utils import get_k_box

ImgPath = '.\\remote_sensing\\images'
XmlPath = '.\\remote_sensing\\annotations'
Label = {'airport': '0', 'bridge': '1', 'harbor': '2', 'ship': '3'}
oW = 0
oH = 0


def get_Label() -> dict:
	"""
	:return: 返回object/label的对应字典
	"""
	return Label


def get_img_list() -> list:
	"""
	:return: 返回图片名称列表
	"""
	img_list = os.listdir(ImgPath)
	return img_list


def get_img_info(img_name: str) -> tuple:
	"""
	:param img_name: 图片名称
	:return: 图片(3 * h * w), bboxs, labels
	"""
	bboxs = []
	img_path = os.path.join(ImgPath, img_name)
	xml_path = os.path.join(XmlPath, img_name[:-4] + '.xml')
	tree = ET.parse(xml_path)
	root = tree.getroot()
	objects = root.findall('object')
	for obj in objects:
		name = obj.find('name').text
		if name in Label.keys():
			info = [item.text for item in obj.find('bndbox')]
			info.append(Label.get(name))
			bboxs.append(','.join(info))
		else:
			print(img_name, name)
	return '.{img} {bboxs}\n'.format(img = img_path, bboxs = ' '.join(bboxs))


def get_img_infos(img_name: str):
	"""
	:param img_name: 图片名称
	:return: 图片(3 * h * w), bboxs, labels
	"""
	global oW, oH
	img_path = os.path.join(ImgPath, img_name)
	img = cv2.imread(img_path)
	H, W, C = img.shape
	if W != oW or H != oH:
		oW = W
		oH = H
		print(img_name, H, W)


def get_bbox():
	xml_list = os.listdir(XmlPath)
	bboxs = []
	for xmls in xml_list:
		xml_path = os.path.join(XmlPath, xmls)
		tree = ET.parse(xml_path)
		root = tree.getroot()
		objects = root.findall('object')
		if len(objects) > 1:
			print(xmls, len(objects))
		for obj in objects:
			xmin, ymin, xmax, ymax = [int(item.text) for item in obj.find('bndbox')]
			bboxs.append([xmax - xmin, ymax - ymin])
	return np.array(bboxs)


def xml2txt():
	"""
	w/h 要满足 >=1, 否则, 转化为 wh互换, 并调整angle: new_angle = (angle + pi/2)%pi
	"""
	all_list = os.listdir(XmlPath)
	all_boxes = []
	for xmlname in all_list:
		bboxs = []
		img_path = os.path.join(ImgPath, xmlname[:-4] + '.jpg')
		xml_path = os.path.join(XmlPath, xmlname)
		tree = ET.parse(xml_path)
		root = tree.getroot()
		objects = root.findall('object')
		for obj in objects:
			name = obj.find('name').text
			if name not in Label.keys():
				print(name)
			bbox = obj.find('bndbox')
			binfo = [int(item.text) for item in bbox]
			x, y = (binfo[0] + binfo[2]) // 2, (binfo[1] + binfo[3]) // 2
			w, h = binfo[2] - binfo[0], binfo[3] - binfo[1]
			binfo = [str(x), str(y), str(w), str(h)]
			binfo.append(Label.get(name))
			bboxs.append(','.join(binfo))
		all_boxes.append('{img} {bboxs}\n'.format(img = img_path.replace('\\', '/'),
		                                          bboxs = ' '.join(bboxs)))
	with open('train.txt', 'w') as f:
		for i in all_boxes:
			f.write(i)


def save_new_image():
	""" 根据train.py, 将图片进行裁剪,并保存 """
	img_path = '.\\images\\img\\{}.jpg'
	txt_path = '.\\images\\train2.txt'
	name = 0
	with open('train.txt') as f:
		lines = f.readlines()

	all_info = []
	for tline in lines:
		line = tline.split()
		rimage = cv2.imread(line[0])
		# (xmi, ymi, xma, yma), box = get_min_box(line, rimage.shape[0:2])
		cnts, boxs = get_k_box(line, rimage.shape[0:2])

		for ks in range(len(cnts)):
			(xmi, ymi, xma, yma), tbox = cnts[ks], boxs[ks]
			img = rimage[ymi:yma + 1, xmi:xma + 1, :]

			# 保存
			img_name = img_path.format(line[0][0:2] + str(name))
			cv2.imwrite(img_name, img)
			name += 1
			box_info = [img_name.replace('\\', '/')]
			for i in tbox:
				box_info.append(','.join(['%.4f' % m for m in i]))
			all_info.append(' '.join(box_info))
	with open(txt_path, 'w') as f:
		f.write('\n'.join(all_info))


if __name__ == '__main__':
	save_new_image()
