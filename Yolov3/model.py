# -*- coding: utf-8 -*-
from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Conv2D, ZeroPadding2D, Concatenate, UpSampling2D, BatchNormalization,
                          GlobalAveragePooling2D, Dense, multiply, add)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Reshape


@wraps(Conv2D)
def DarkConv2D(*args, **kwargs):
	"""darknet Conv2D网络, 使用L2正则, strides为2时减半图像边长像素"""
	net_kwargs = kwargs
	#  使用L2正则
	net_kwargs['kernel_regularizer'] = l2(5e-4)
	# 若步长为2, 则padding为0, 使得图像边长减半, 起到pool的作用
	net_kwargs['padding'] = \
		'valid' if kwargs.get('strides', None) == (2, 2) else 'same'
	return Conv2D(*args, **net_kwargs)


def SE(inputs, ratio = 16):
	# SE分支
	init = inputs
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	filters = init._keras_shape[channel_axis]
	se_shape = (1, 1, filters)

	se = GlobalAveragePooling2D()(init)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(se)
	se = Dense(filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = False)(se)

	# multiply: 逐元素相乘
	x = multiply([init, se])
	return x


def DarkConv2D_BN_Leaky(inputs, *args, **kwargs):
	"""DBL网络, 即使用BN, LeakyReLU的DarkConv2D网络"""
	net_kwargs = kwargs
	# 不使用偏置参数
	net_kwargs['use_bias'] = False
	net = DarkConv2D(*args, **net_kwargs)(inputs)
	net = BatchNormalization()(net)
	net = LeakyReLU(alpha = 0.1)(net)
	return net


def resblock_body(x, num_filters, num_blocks):
	"""
	返回Resblock网络, 即zeropadding + DBL + num_blocks个res网络
	res网络: 两个DBL的残差和
	:param x: 输入网络
	:param num_filters: DBL的filter_num
	:param num_blocks: 残差网络重复的次数
	:return:
	"""
	# Darknet uses left and top padding instead of 'same' mode
	x = ZeroPadding2D(((1, 0), (1, 0)))(x)
	x = DarkConv2D_BN_Leaky(x, num_filters, (3, 3), strides = (2, 2))  # 2倍下采样
	#  num_blocks个残差
	for num in range(num_blocks):
		y = DarkConv2D_BN_Leaky(x, num_filters // 2, (1, 1))
		y = DarkConv2D_BN_Leaky(y, num_filters, (3, 3))
		x = add([y, x])
	return x


def dark_body(x):
	"""返回darhnet的主要网络"""
	x = DarkConv2D_BN_Leaky(x, 32, (3, 3))
	x = resblock_body(x, 64, 1)
	x = resblock_body(x, 128, 2)
	x = resblock_body(x, 256, 8)
	y3 = x
	x = resblock_body(x, 512, 8)
	y2 = x
	x = resblock_body(x, 1024, 4)
	y1 = x
	return y3, y2, y1


def last_layer(x, num_filters, out_filters):
	"""返回最后的 (5个DBL) + (DBL + darkCond2D)"""
	x = DarkConv2D_BN_Leaky(x, num_filters, (1, 1))
	x = DarkConv2D_BN_Leaky(x, num_filters * 2, (3, 3))
	x = DarkConv2D_BN_Leaky(x, num_filters, (1, 1))
	x = DarkConv2D_BN_Leaky(x, num_filters * 2, (3, 3))
	x = DarkConv2D_BN_Leaky(x, num_filters, (1, 1))
	y = DarkConv2D_BN_Leaky(x, num_filters * 2, (3, 3))
	y = DarkConv2D(out_filters, (1, 1))(y)
	return x, y


def yolo_body(inputs, num_anchors, num_classes):
	"""
	返回yolo主体
	:param inputs: 输入层, hw
	:param num_anchors: 每个grid的anchors数量
	:param num_classes: 预测的class数量
	:return:
	"""

	# 使用 Model而不是直接 dark_body(inputs): 方便取中间层的输出
	ty3, ty2, ty1 = dark_body(inputs)
	# 第一个输出
	x, y1 = last_layer(ty1, 512, num_anchors * (num_classes + 5))
	# 第二个输出
	# 上一层的x + DBL + 上采用层
	x = DarkConv2D_BN_Leaky(x, 256, (1, 1))
	x = UpSampling2D(2)(x)
	# 直接串联两个输出
	x = Concatenate()([x, ty2])
	x, y2 = last_layer(x, 256, num_anchors * (num_classes + 5))

	# 第三个输出
	x = DarkConv2D_BN_Leaky(x, 128, (1, 1))
	x = UpSampling2D(2)(x)
	x = Concatenate()([x, ty3])
	_, y3 = last_layer(x, 128, num_anchors * (num_classes + 5))

	return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss = False):
	"""
	x y w h c a ain*4 num_class
	根据 feature map 输出相对于 输入图像 的bbox的坐标, 置信度, obj概率
	box_xy: 相对于特征图的比例位置(和在输入图像中相对于输入图像的比例位置效果是相同的)
	box_wh: 相对于输入图像的比例
	:param feats: 特征图feature map, (batch_size, g_y, g_x, num_anchor*(num_class + 10)) (即网络的输出 y1 y2 y3)
	:param anchors: 对该特征图的 anchor
	:param num_classes:预测的obj的数量
	:param input_shape: 输入图像的shape, wh
	:param calc_loss:是否是计算loss, 是则返回grid, feats, box_xy, box_wh
	:return: feature map中的信息(与y_true对应)
	"""
	num_anchors = len(anchors)
	# 转置张量, 方便后续运算 [batch, height, width, num_anchors, box_params]
	anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
	# quarter = np.pi / 4  # pi/4

	# 特征图的 H, W
	# 这里特征图一个像素对应原图中的一个cell,
	# 若输入:[16*16], 输出(特征图):[4*4],
	# 则特征图中一个像素[0, 0]对应输入图像中[0:4, 0:4]的cell
	grid_shape = K.shape(feats)[1:3]

	# tile:平铺扩展张量
	# grid_y.shape: [grid_shape[0], grid_shape[1], 1, 1],
	# grid_y值为该cell相对于左上角的垂直y距离(即像素距离)
	grid_y = K.tile(K.reshape(K.arange(0, grid_shape[0]), [-1, 1, 1, 1]),
	                [1, grid_shape[1], 1, 1])  # (x, y, 1, 1) x维数据为0, 1, 2..., y维数据均相同,为第一项
	grid_x = K.tile(K.reshape(K.arange(0, grid_shape[1]), [1, -1, 1, 1]),
	                [grid_shape[0], 1, 1, 1])  # (x, y, 1, 1) y维数据为0, 1, 2..., x维数据均相同,为第一项

	# concatenate:指定轴连接张量, 默认第一个维度
	# cast:将张量转换到不同的 dtype
	grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

	feats = K.reshape(
		feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

	# Adjust preditions to each spatial grid point and anchor size.
	# [...]: 前面所有维度
	# box_xy: 相对于特征图的比例位置
	box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
	# [::] 第一个维度的start:stop:stride,切片操作
	# [::-1] 相当于每个维度反序所有元素
	# (w,h)是bbox相对于整个图片的比例
	# [..., 0:1]这样取值维度不会发生改变, [..., 0]会减少一维
	box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
	box_confidence = K.sigmoid(feats[..., 4:5])
	box_class_probs = K.sigmoid(feats[..., 5:])

	if calc_loss:
		return grid, feats, box_xy, box_wh

	return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
	"""
	将 bbox 回放到原来的image上去 (按比例放大)
	:param angle_info: (..., 5), 角度, 角度范围
	:param box_xy:
	:param box_wh:
	:param input_shape: 输入图像尺寸, hw
	:param image_shape: 原图尺寸, hw
	:return:
	"""
	input_shape = input_shape[..., ::-1]
	image_shape = image_shape[..., ::-1]

	input_shape = K.cast(input_shape, K.dtype(box_xy))
	image_shape = K.cast(image_shape, K.dtype(box_xy))
	# round: 元素级地四舍五入到最接近的整数
	# new_shape: 按原比例缩放后的shape
	new_shape = K.round(image_shape * K.min(input_shape / image_shape))
	# offest: 上左空白处占input_shape的比例
	offset = (input_shape - new_shape) / 2.0 / input_shape
	scale = input_shape / new_shape
	# 计算出yxwh占image的比例, * image_shape后变为真实位置
	box_xy = (box_xy - offset) * scale * image_shape
	box_wh *= scale * image_shape

	boxes = K.concatenate([box_xy, box_wh], axis = -1)
	return boxes


def yolo_boxes_and_scores(feats, anchors, num_class, input_shape, image_shape):
	""" 处理网络的输出, 返回实际位置bbox, 相应分数"""
	box_xy, box_wh, box_confidence, box_class_probs = \
		yolo_head(feats, anchors, num_class, input_shape)

	boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
	boxes = K.reshape(boxes, [-1, 4])

	box_scores = box_confidence * box_class_probs
	box_scores = K.reshape(box_scores, [-1, num_class])
	# boxes与box_scores是一一对应的
	return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              score_threshold = 0.6):
	"""
	获取最终输出
	:param yolo_outputs: 网络输出
	:param anchors: 所有的anchors
	:param num_classes: obj种类数量
	:param image_shape: 原图尺寸, tensor
	:param score_threshold: 分数阈值
	:return:
	"""
	num_layers = len(yolo_outputs)
	each = len(anchors) // 3
	# anchor_mask:[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	anchor_mask = [list(range(i * each, (i + 1) * each)) for i in range(2, -1, -1)]
	input_shape = K.shape(yolo_outputs[0])[1:3] * 32
	boxes = []
	box_scores = []
	for l in range(num_layers):
		_boxes, _box_scores = \
			yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
			                      num_classes, input_shape, image_shape)
		boxes.append(_boxes)
		box_scores.append(_box_scores)
	boxes = K.concatenate(boxes, axis = 0)
	box_scores = K.concatenate(box_scores, axis = 0)

	mask = (box_scores >= score_threshold)
	boxes_ = []  # 顺时针的四个坐标 (..., 8)
	scores_ = []
	classes_ = []
	for c in range(num_classes):
		# 去除掉分数过低的bbox, 没有使用NMS.
		# boolean_mask: 从boxes中取出mask为True的值
		class_boxes = tf.boolean_mask(boxes, mask[:, c])
		class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

		classes = K.ones_like(class_box_scores, 'int32') * c
		boxes_.append(class_boxes)
		scores_.append(class_box_scores)
		classes_.append(classes)
	boxes_ = K.concatenate(boxes_, axis = 0)
	scores_ = K.concatenate(scores_, axis = 0)
	classes_ = K.concatenate(classes_, axis = 0)

	# 没有经过NMS, boxes为(x, y, w, h, angle)
	return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes: np.ndarray, input_shape, anchors, num_classes):
	"""
	true_boxes: x y w h angle class, angle:顺时针方向, 范围:[0, pi]
	即将原始box 信息 转换成训练时输入的格式, 输出用于计算loss
	返回[ m*13*13*3*(10+num_class), m*26*26*3*(10+num_class), m*52*52*3*(10+num_class)]
	:param true_boxes: (m, T, 6), 表示m个图像  每个图像T个box, xywh angle class_id
	:param input_shape: hw, 必须是32的倍数
	:param anchors: (N, 2), 共N种, wh
	:param num_classes: class数量
	:return: [ m*13*13*3*(10+num_class), m*26*26*3*(10+num_class), m*52*52*3*(10+num_class)]
	"""
	# true_boxes[, 5] 为class id, 要小于class数量, 验证正确性
	assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
	# 每个feature map分 3个 anchor
	# quarter = np.pi / 4  # pi/4
	num_layers = 3  # default setting, 3个输出
	each = len(anchors) // 3
	# [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	anchor_mask = [list(range(i * each, (i + 1) * each)) for i in [2, 1, 0]]

	input_shape = np.array(input_shape, dtype = 'int32')
	boxes_xy = true_boxes[..., 0:2]  # (m, T, 2)
	boxes_wh = true_boxes[..., 2:4]  # (m, T, 2)
	true_boxes = np.array(true_boxes, dtype = 'float32')  # (m, T, 5)
	# boxs_angle = true_boxes[..., 4:5]
	# boxs_angle[boxs_angle > (np.pi / 2)] -= np.pi  # 转化为顺时针, 取值[-pi/2, pi/2]

	# 转换成相对于整张图片的比例 input_shape是hw, 要反序为wh
	true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
	true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

	m = true_boxes.shape[0]

	# grid_shape [13,13]   [26,26]  [52,52]
	grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

	# [ m*13*13*3*(5+num_class), m*26*26*3*(5+num_class), m*52*52*3*(5+num_class)]
	y_true = [np.zeros(
		(m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
		dtype = 'float32') for l in range(num_layers)]

	# Expand dim to apply broadcasting.
	# expand_dims: 在axis(=0)维扩展数组的形状
	anchors = np.expand_dims(anchors, 0)  # (1, N, 2)
	# 以网格中心为原点（即网格中心坐标为0,0）,计算出anchor 右下角坐标
	anchor_maxes = anchors / 2.0
	anchor_mins = -anchor_maxes
	# valid_mask中flase为异常数据(wh小于0了)
	valid_mask = boxes_wh[..., 0] > 0

	for b in range(m):
		# 选取正常的数据.
		wh = boxes_wh[b, valid_mask[b]]  # (T, 2)
		if len(wh) == 0:
			continue
		# Expand dim to apply broadcasting.
		wh = np.expand_dims(wh, -2)  # (T, 1, 2)
		box_maxes = wh / 2.
		box_mins = -box_maxes

		# 计算 ground_true与anchor box的交并比, 这里不用计算斜矩形IOU, 只需找到大小最相近的anchor即可
		# intersect_area: box 与 anchor 重合的面积
		# maximum: 逐位比较(将每个anchor与每个box比较)  (T, 1, 2) 与 (1, N, 2)
		intersect_mins = np.maximum(box_mins, anchor_mins)  # (T, N, 2)
		intersect_maxes = np.minimum(box_maxes, anchor_maxes)
		intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
		intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # (T, N)
		box_area = wh[..., 0] * wh[..., 1]  # (T, 1)
		anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, N)
		iou = intersect_area / (box_area + anchor_area - intersect_area)

		# Find best anchor for each true box
		best_anchor = np.argmax(iou, axis = -1)  # iou最大的anchor的下标, (T, 1)

		# enumerate: 同时列出数据下标和数据
		for t, n in enumerate(best_anchor):  # 在所有的最优下标中
			for l in range(num_layers):  # 在每个layer的anchor中
				if n in anchor_mask[l]:
					# np.floor 返回不大于输入参数的最大整数
					# i, j: 转换到feature map上的坐标 (左上角为原点)
					i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
					j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
					k = anchor_mask[l].index(n)  # anchor所在的索引
					c = true_boxes[b, t, 4].astype('int32')  # 种类索引
					# index = best_angle_index[t, 0]
					# xywh(相对于整张图片的比例), 置信度, angle, angle_index, class种类
					y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
					y_true[l][b, j, i, k, 4] = 1
					y_true[l][b, j, i, k, 5 + c] = 1
	# (3, m, g_y, g_x, num_anchor, 10+num_class), m为图片数量(batch_size)
	return y_true


def box_iou(b1, b2):
	"""
	计算两组Box的IOU, 使用 iou * min(cos(a1 - a2) * w1/h1 * w2/h2), 1)
	:param b1: pre, tensor, shape=(i1,...,iN, 4), xywh
	:param b2: true, tensor, shape=(j, 4), xywh
	:return: tensor, shape=(i1,...,iN, j)
	"""

	# Expand dim to apply broadcasting.
	b1 = K.expand_dims(b1, -2)  # (i1~in, 1, 4)
	b1_xy = b1[..., 0:2]
	b1_wh = b1[..., 2:4]
	b1_wh_half = b1_wh / 2.0
	b1_mins = b1_xy - b1_wh_half
	b1_maxes = b1_xy + b1_wh_half

	# Expand dim to apply broadcasting.
	b2 = K.expand_dims(b2, 0)  # (1, j, 4)
	b2_xy = b2[..., 0:2]
	b2_wh = b2[..., 2:4]
	b2_wh_half = b2_wh / 2.0
	b2_mins = b2_xy - b2_wh_half
	b2_maxes = b2_xy + b2_wh_half

	intersect_mins = K.maximum(b1_mins, b2_mins)
	intersect_maxes = K.minimum(b1_maxes, b2_maxes)
	intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.0)
	intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
	b1_area = b1_wh[..., 0] * b1_wh[..., 1]
	b2_area = b2_wh[..., 0] * b2_wh[..., 1]
	iou = intersect_area / (b1_area + b2_area - intersect_area)

	return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh = 0.5, print_loss = False):
	"""
	计算loss tensor
	:param args: (yolo_output_list([y1, y2, y3]), y_true(list of array)
	:param anchors: array, shape=(N, 2), wh
	:param num_classes:
	:param ignore_thresh: the iou threshold whether to ignore object confidence loss
	:param print_loss:
	:return: tensor, shape=(1,)
	"""
	# num_layers = len(anchors) // 3  # default setting
	num_layers = 3  # 3个输出 default setting
	# shape: 3* batch* feature_map_x* feature_map_y* 3* (5+num_class)
	yolo_outputs = [tf.convert_to_tensor(i) for i in args[:num_layers]]
	y_true = [tf.convert_to_tensor(i) for i in args[num_layers:]]
	each = len(anchors) // 3  # 每个输出有each个anchor
	anchor_mask = [list(range(i * each, (i + 1) * each)) for i in [2, 1, 0]]
	input_shape = K.cast(
		K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
	# [13*13, 36*36, 52*52]
	grid_shapes = [K.cast(
		K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
	loss = 0
	m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
	mf = K.cast(m, K.dtype(yolo_outputs[0]))

	for l in range(num_layers):
		# 真实置信度(有为1,无为0)
		object_mask = y_true[l][..., 4:5]
		# obj种类
		true_class_probs = y_true[l][..., 5:]

		# 返回grid, feats, box_xy, box_wh,
		# raw_pred即feats格式与y_true相同, 为网络的输出, 未经过转换
		# pred_xy, pred_wh 均为相对于特征图(也相当于输入图片)的比例
		grid, raw_pred, pred_xy, pred_wh = \
			yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss = True)
		pred_box = K.concatenate([pred_xy, pred_wh])  # (m, g_y, g_x, 3, xywh)

		# Darknet raw box to calculate loss. 即计算bx, by, bw, bh
		raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
		raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
		# switch(a, b, c): 根据标量值在两个操作之间切换, 即 = b if a else c
		# 将-inf替换为0 (若object_mask为0, 则raw_true_wh为-inf, 然后用zeros_like代替
		raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

		# 2 - w*h, w*h越小, box_loss_scale越大
		# w*h越小的anchor, iou必然就小, 导致“存在物体”的置信度就越小. 也就是object_mask越小
		box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

		# Find ignore mask, iterate over each of batch.
		# tf.TensorArray: 生成一个动态数组
		ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size = 1, dynamic_size = True)
		object_mask_bool = K.cast(object_mask, 'bool')

		def loop_body(b, ignore_masks):
			"""b: 第b（即mini_batch_size）个图像, """
			# boolean_mask(a, b):a中抽取索引为b的tensor组成在一起
			true_box = tf.boolean_mask(
				y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])  # 置信度为1的grid, (num_true_anchor, )
			iou = box_iou(pred_box[b], true_box)
			best_iou = K.max(iou, axis = -1)
			# 如果一张图片的最大iou 都小于阈值 认为这张图片没有目标
			ignore_masks = ignore_masks.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
			return b + 1, ignore_masks

		# while_loop(cond,body,loop_vars): Repeat `body` while the condition `cond` is true
		# m: batch size
		_, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

		# 如果某个anchor不负责预测GT，且该anchor预测的框与图中所有GT的IOU都小于某个阈值，则让它预测背景，
		# 如果大于阈值则不参与损失计算
		ignore_mask = ignore_mask.stack()
		ignore_mask = K.expand_dims(ignore_mask, -1)

		# K.binary_crossentropy is helpful to avoid exp overflow. 输出张量与目标张量之间的二进制交叉熵。
		xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
		                                                               from_logits = True)
		# square 元素级的平方操作
		wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
		# 正样本 + 负样本(预测背景)
		confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits = True) + \
		                  (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
		                                                            from_logits = True) * ignore_mask

		class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits = True)

		xy_loss = K.sum(xy_loss) / mf
		wh_loss = K.sum(wh_loss) / mf
		confidence_loss = K.sum(confidence_loss) / mf
		class_loss = K.sum(class_loss) / mf
		loss += xy_loss + wh_loss + confidence_loss + class_loss
		if print_loss:
			loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss,
			                       class_loss, K.sum(ignore_mask)],
			                message = 'loss: ')
	return loss
