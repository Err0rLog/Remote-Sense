# -*- coding: utf-8 -*-
"""
 @Time    : 2019/4/29 下午 01:39
 @Author  : Err0r log
"""
import numpy as np
import collections
import pprint
import matplotlib.pyplot as plt

_W, _H = 400, 400
_mins = 10


def get_new_center(data, index):
	"""
	返回这些点的中心点索引, 时候满足小于[416, 416]
	:param data: ndarray, (N, 5)
	:param index: ndarray, (k, )
	:return: 中心点索引, 是返回0, 不是返回1
	"""
	global _W, _H, _mins
	_data = data[index]
	half_w = _data[..., 2]
	half_h = _data[..., 3]
	middle_x = np.sum(_data[..., 0]) / len(index)
	middle_y = np.sum(_data[..., 1]) / len(index)
	_ndata = np.array(_data[..., 0:2])

	# x和y 小于终点的均减去w的一半, 大于终点的均加上w的一半, y类同
	_g = np.ones_like(_data[..., 0])
	_g[_data[..., 0] < middle_x] = -1
	_ndata[..., 0] += _g * half_w

	_g = np.ones_like(_data[..., 1])
	_g[_data[..., 1] < middle_y] = -1
	_ndata[..., 1] += _g * half_h

	# 获取中心点
	_xy = np.sum(_ndata, axis = -2) / len(index)
	new = index[np.argmin(np.sum(np.abs(_ndata - _xy), -1))]

	# 是否满足范围在[416, 416]内
	min_wh = np.min(data[..., 2:4], -2)
	cx = max(min_wh[0] // _mins, 1)
	cy = max(min_wh[1] // _mins, 1)
	axis_xy = np.max(_ndata, -2) - np.min(_ndata, -2)
	# 0表示满足要求
	nook = 0 if (axis_xy <= [_W * cx, _H * cy]).all() else 1
	# print(axis_xy, axis_xy <= [_W * cx, _H * cy], nook)

	return new, nook


def farthest(data, centers):
	"""
	返回距离多个centers最远的point的index
	:param data: ndarray, (N, 5): xywh
	:param centers: ndarray, (k, ): index
	:return:
	"""
	k = len(centers)
	# dis = np.zeros(len(data))
	center_xy = data[centers, 0:2]  # (k, 2)
	_data = np.expand_dims(data[..., 0:2], -2)  # (N, 1, 2)
	temp_xy = np.abs(_data - center_xy)  # (N, k, 2)
	dis = np.max(temp_xy, -1) * (np.random.rand(k) * 0.8 + 0.2)  # (N, k)
	dis = np.sum(dis, -1)

	d = np.argsort(dis)
	p = -1
	try:
		while d[p] in centers:
			p -= 1
	except IndexError as e:
		print(e)
		raise IndexError('图像分割失败')
	return d[p]


def get_centers(data: np.ndarray, k = 1):
	"""
	返回预先初始化的几个中心索引, (k, )
	:param data: ndarray, (N, 5): xywha
	:param k: k个中心
	:return:
	"""
	global _W, _H, _mins
	N = len(data)
	# min_wh = np.min(data[..., 2:4], -2)
	# max_xy = np.max(data[..., 0:2], -2) - np.min(data[..., 0:2], -2)
	# W = _W * (min_wh[0] // _mins) if min_wh[0] > _mins else _W
	# H = _H * (min_wh[1] // _mins) if min_wh[1] > _mins else _H
	# min_k = (np.floor(max_xy[0] / W) * np.floor(max_xy[1] / H)).astype(int)

	centers = np.zeros(k, dtype = np.int)
	centers[0] = np.random.randint(0, N)

	for i in range(1, k):
		centers[i] = farthest(data, centers[0:i])

	return centers


def show_point(data, point, centers):
	""" 将kmeans的结果绘制出来 """
	marker = ['o', 'v', '^', '>', '<', '1', '2', '3', '4', 's', 'p', '*', 'h', '+', 'x', 'd']
	# pprint.pprint(collections.Counter(point))
	for m in range(len(centers)):
		x0 = data[point == m]
		plt.scatter(x0[:, 0], x0[:, 1], marker = marker[m % len(marker)])
	ax = plt.gca()
	ax.xaxis.set_ticks_position('top')
	ax.invert_yaxis()
	plt.title(str(len(centers)))
	plt.show()


# 主程序
def kmeans_box(data):
	"""

	:param data: ndarray, (N, 4)
	:return: ((N,), num_box)
	"""
	global _W, _H, _mins
	N = len(data)
	# axis = np.ceil(N / 5) if N < 10 else 2
	axis = 0

	centers = get_centers(data)

	_data = np.expand_dims(data[..., 0:2], -2)  # (N, 1, 2)
	_centers = data[centers, 0:2]  # (k, 2)
	_dis = np.max(np.abs(_data - _centers), -1)  # (N, k)
	_dis_point = np.argmin(_dis, -1)  # (N, ), 值为所属的距离最小的center的index
	# show_point(data, _dis_point, centers)

	times = 0
	while True:
		times += 1
		# 强制洗牌
		if times % 20 == 0:
			centers = get_centers(data, len(centers))
			# 更新每个点的范围
			_centers = data[centers, 0:2]  # (k, 2)
			_dis = np.max(np.abs(_data - _centers), -1)  # (N, k)
			_dis_point = np.argmin(_dis, -1)  # (N, ), 值为所属的距离最小的center的index

		is_nook = np.ones(len(centers))
		for m in range(len(centers)):
			min_index = np.array(np.where(_dis_point == m)[0])
			centers[m], is_nook[m] = get_new_center(data, min_index)

		# 分类完成
		if not np.sum(is_nook):
			return _dis_point, len(centers)

		_centers = data[centers, 0:2]  # (k, 2)
		_dis = np.max(np.abs(_data - _centers), -1)  # (N, k)
		_new_dis_point = np.argmin(_dis, -1)  # (N, ), 值为所属的距离最小的center的index

		if np.sum(np.abs(_new_dis_point - _dis_point)) <= axis and np.random.random() > 0.7:
			# 添加一个中心点
			centers = np.append(centers, farthest(data, centers))
			# 每增加10个点就重新洗牌
			if len(centers) % 10 == 0:
				print('洗牌', len(centers))
				centers = get_centers(data, len(centers))
			# 更新每个点的范围
			_centers = data[centers, 0:2]  # (k, 2)
			_dis = np.max(np.abs(_data - _centers), -1)  # (N, k)
			_new_dis_point = np.argmin(_dis, -1)  # (N, ), 值为所属的距离最小的center的index
			if len(centers) > N:
				print('None')
				return _new_dis_point, None
		_dis_point = _new_dis_point
		# show_point(data, _dis_point, centers)
