# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def nms(boxes, scores, thresh):
	if len(boxes) == 0:
		return []
	boxes = boxes.astype("float")

	pick = []
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(scores)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		overlap = (w * h) / area[idxs[:last]]

		# 删除所有重叠率大于阈值的边界框
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))
	return pick


def letterbox_image(image: np.ndarray, size) -> np.ndarray:
	"""reshape image to size"""
	ih, iw = image.shape[0:2]
	h, w = size

	scale = min(w / iw, h / ih)
	nw = int(iw * scale)
	nh = int(ih * scale)
	dx = (w - nw) // 2
	dy = (h - nh) // 2
	image = cv2.resize(image, (nw, nh))
	# 周围填充
	new_image = cv2.copyMakeBorder(image,
	                               dy, h - dy - nh, dx, w - dx - nw,
	                               cv2.BORDER_CONSTANT,
	                               value = (128, 128, 128))
	return new_image


def rrand(a = 0., b = 1.):
	"""返回a-b之间的随机数"""
	# np.random.rand(): 返回一个或一组服从“0~1”均匀分布的随机样本值
	return np.random.rand() * (b - a) + a


def wh2xy(box):
	"""
	[..., x, y, w, h] to [..., xy1, xy2, xy3, xy4]
	angle: 顺时针方向的弧度, 取值[-pi/2, pi/2]
	:param box:
	:param train_flg: 训练时的angle为相对于x负半轴,顺时针方向的弧度, 取值[0, pi]
	:param radian: 是否为弧度, False为角度
	:return:
	"""
	xmin = box[..., 0:1] - box[..., 2:3] // 2
	xmax = box[..., 0:1] + box[..., 2:3] // 2
	ymin = box[..., 1:2] - box[..., 3:4] // 2
	ymax = box[..., 1:2] + box[..., 3:4] // 2
	boxes = np.concatenate([xmin, ymin, xmax, ymax], -1)
	return boxes


def draw_box(image, boxes, scores, classes, colors, class_names):
	"""
	在图像上绘制预测出的斜矩形框
	:param image: ndarray, (h, w, 3), cv2读取的图像
	:param boxes: ndarray, (N, 4), N个box, xmin, ymin, xmax, ymax
	:param scores: ndarray, (N, 1), 每个box对应的分数
	:param classes: ndarray, (N, 1), 每个box对用的对象, 值的取值为(0, 1, 2, 3), 对应四种识别的对象
	:param colors: list, (N, 3), 每种class的color
	:param class_names: list, (N, ), class对应名称
	:return: 绘制好box的image
	"""
	# boxes: [N, 4, 1, 2], out_scores: [N, 1], out_classes: [N, 1]
	fontScale = 0.7  # 字体库大小的倍数
	thickness = 2  # 文字的粗细
	image_w = image.shape[1]

	for i, c in enumerate(classes):
		predicted_class = class_names[c]
		box = boxes[i]
		score = scores[i]

		label = '{} {:.2f}'.format(predicted_class, score)
		bx, by = box[0:2]  # 最高的那个点
		tx, ty = cv2.getTextSize(label,
		                         fontFace = cv2.FONT_HERSHEY_SIMPLEX,
		                         fontScale = fontScale,  # 字体库大小的倍数
		                         thickness = thickness)  # 划线的粗细
		# font_size, see https://blog.csdn.net/u010970514/article/details/84075776
		tx, ty, tty = tx[0], tx[1], tx[1] + ty
		t_left = bx if bx + tx < image_w else image_w - tx - 5
		t_buttom = by - 1 if by - 1 > tty else by + tty + 1

		# 画框
		cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[c], 2)

		# 画文字背景
		# image = cv2.rectangle(image, (t_left, t_top), (t_right, t_buttom), colors[c], thickness = -1)
		# 文字
		cv2.putText(image, label, (t_left, t_buttom),
		            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
		            fontScale = fontScale,
		            color = colors[c],
		            thickness = thickness)
	return image


def get_k_box(line, shape):
	"""
	返回k个聚类后的box
	([k, 4], [k, n, 6]): 4: xmin, ymin, xmax, ymax; n: n个box
	"""
	from boxkmeans import kmeans_box
	h, w = shape
	box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
	data, num = kmeans_box(box[..., 0:4])
	xy = wh2xy(box)

	imgs = []
	boxs = []
	for i in range(num):
		index = np.array(np.where(data == i)).flatten()
		# print(index, xy[index])
		xmin, ymin = np.min(xy[index][..., 0:2], -2)
		xmax, ymax = np.max(xy[index][..., 2:4], -2)
		# 调整范围
		if xmax - xmin + 1 >= 416:
			xmin = (xmin - 15) if xmin > 15 else 0
			xmax = (xmax + 15) if (w - xmax) > 15 else w
		else:
			dx = (416 - xmax + xmin) / 2
			if xmax + dx > w:
				xmax = w
				xmin = max(0, xmax - 415)
			else:
				xmin = xmin - dx if xmin >= dx else 0
				xmax = min(w, xmin + 415)

		if ymax - ymin + 1 >= 416:
			ymin = (ymin - 15) if ymin > 15 else 0
			ymax = (ymax + 15) if (h - ymax) > 15 else h
		else:
			dy = (416 - ymax + ymin) / 2
			if ymax + dy > h:
				ymax = h
				ymin = max(0, ymax - 415)
			else:
				ymin = ymin - dy if ymin >= dy else 0
				ymax = min(h, ymin + 415)

		# 使图片方一点
		if xmax - xmin - 40 > ymax - ymin:
			dy = (xmax - xmin - (ymax - ymin)) // 2
			ymax = min(h, ymax + dy)
			ymin = max(0, ymin - dy)

		elif ymax - ymin - 40 > xmax - xmin:
			dx = (ymax - ymin - (xmax - xmin)) // 2
			xmax = min(w, xmax + dx)
			xmin = max(0, xmin - dx)

		imgs.append(np.floor([xmin, ymin, xmax, ymax]).astype(np.int))
		boxs.append(box[index] - [xmin, ymin, 0, 0, 0])

	# TODO:只包含了部分box, 待修改===============================
	return imgs, boxs


def get_min_box(line, shape):
	"""
	获取包含所有对象的最小包围框
	:param shape:
	:param line:
	:return:([xmin, ymin, xmax, ymax]), (..., x, y, w, h, angle, class)
	"""
	h, w = shape
	box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
	xy_box = wh2xy(box, train_flg = True)
	# 获取最小包围框
	xmin = np.min(xy_box[..., 0])
	ymin = np.min(xy_box[..., 1])
	xmax = np.max(xy_box[..., 0])
	ymax = np.max(xy_box[..., 1])
	# 调整范围
	if xmax - xmin + 1 >= 416:
		xmin = (xmin - 15) if xmin > 15 else 0
		xmax = (xmax + 15) if (w - xmax) > 15 else w
	else:
		dx = (416 - xmax + xmin) / 2
		if xmax + dx > w:
			xmax = w
			xmin = max(0, xmax - 415)
		else:
			xmin = xmin - dx if xmin >= dx else 0
			xmax = min(w, xmin + 415)

	if ymax - ymin >= 416:
		ymin = (ymin - 15) if ymin > 15 else 0
		ymax = (ymax + 15) if (h - ymax) > 15 else h
	else:
		dy = (416 - ymax + ymin) / 2
		if ymax + dy > h:
			ymax = h
			ymin = max(0, ymax - 415)
		else:
			ymin = ymin - dy if ymin >= dy else 0
			ymax = min(h, ymin + 415)

	# 计算新的box的位置
	box[..., 0] -= xmin
	box[..., 1] -= ymin

	return np.floor([xmin, ymin, xmax, ymax]).astype(np.int), box


def get_random_data(annotation_line, input_shape,
                    max_boxes = 20, hue = 0.1,
                    sat = 1.5, val = 1.5,
                    random = True, proc_img = True):
	"""
	返回image, boxs
	annotation_line: train.txt读取的每一行文本:
		格式:img_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
	input_shape: 网络输入图像的尺寸
	random: 是否随机化处理(对比度,bbox位置(微调)...)
	max_boxes: 一张图像最多计算的bbox数量
	"""
	line = annotation_line.split()
	image = cv2.imread(line[0])

	box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
	ih, iw = image.shape[0:2]
	h, w = input_shape

	if not random:
		# resize image
		# 等比例缩放后, 放在输入图像的中间位置, 四周用灰色(128*3)填充
		scale = min(w / iw, h / ih)
		nw = int(iw * scale)
		nh = int(ih * scale)
		dx = (w - nw) // 2
		dy = (h - nh) // 2
		image_data = 0
		if proc_img:
			image_data = cv2.copyMakeBorder(cv2.resize(image, (nw, nh)),
			                                dy, h - dy - nh, dx, w - dx - nw,
			                                cv2.BORDER_CONSTANT,
			                                value = (128, 128, 128))

		# correct boxes
		box_data = np.zeros((max_boxes, 5))
		if len(box) > 0:
			np.random.shuffle(box)
			if len(box) > max_boxes:
				box = box[:max_boxes]
			box[:, [0]] = box[:, [0]] * nw / iw + dx
			box[:, [1]] = box[:, [1]] * nh / ih + dy
			box[:, [2]] = box[:, [2]] * nw / iw
			box[:, [3]] = box[:, [3]] * nh / ih
			box_data[:len(box)] = box

		return image_data, box_data

	# map(f, list) 把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
	# box: (x, y, w, h, angle, class)

	# resize image
	scale = min(w / iw, h / ih) * rrand(0.9, 1)
	nw = int(iw * scale)
	nh = int(ih * scale)

	image = cv2.resize(image, (nw, nh))

	# place image
	dx = int(rrand(0, w - nw))
	dy = int(rrand(0, h - nh))
	image = cv2.copyMakeBorder(image,
	                           dy, h - dy - nh, dx, w - dx - nw,
	                           cv2.BORDER_CONSTANT,
	                           value = (128, 128, 128))

	# flip image or not
	flip = rrand() < 0.5
	if flip:
		image = cv2.flip(image, 1)

	# distort image
	hue = rrand(-hue, hue)
	sat = rrand(1, sat) if rrand() < 0.5 else 1 / rrand(1, sat)
	val = rrand(1, val) if rrand() < 0.5 else 1 / rrand(1, val)

	x = rgb_to_hsv(image / 255.0)
	# 一波随机化骚操作
	x[..., 0] += hue
	x[..., 0][x[..., 0] > 1] -= 1
	x[..., 0][x[..., 0] < 0] += 1
	x[..., 1] *= sat
	x[..., 2] *= val
	x[x > 1] = 1
	x[x < 0] = 0
	image = hsv_to_rgb(x)  # numpy array, 0 to 1

	# correct boxes
	box_data = np.zeros((max_boxes, 5))

	# 压缩后, 角度也会发生变化, 除非是等比例压缩
	if len(box) > 0:
		np.random.shuffle(box)
		box[:, [0]] = box[:, [0]] * nw / iw + dx
		box[:, [1]] = box[:, [1]] * nh / ih + dy
		box[:, [2]] = box[:, [2]] * nw / iw
		box[:, [3]] = box[:, [3]] * nh / ih

		if flip:  # 左右翻转
			box[:, [0]] = w - box[:, [0]]
		# box[:, [4]] = np.pi - box[:, [4]]  # 角度

		# 出界情况, 未修改=================================================
		# box[:, 0:2][box[:, 0:2] < 0] = 0
		# box[:, 2][box[:, 2] > w] = w
		# box[:, 3][box[:, 3] > h] = h
		# box_w = box[:, 2] - box[:, 0]
		# box_h = box[:, 3] - box[:, 1]
		# box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

		if len(box) > max_boxes:
			box = box[:max_boxes]

		box_data[:len(box)] = box

	return image, box_data


def segmentation(image, img_size, proportion = 8):
	"""
	分割图像
	:param image:被分割图像
	:param img_size: 输入size
	:param proportion: 重叠面积占比
	:return: list, (N, 4), xmin, ymin, xmax, ymax
	"""
	H, W, _ = image.shape
	h, w = img_size
	mh, mw = h // proportion, w // proportion
	size = []
	if H <= (h + mh) and W <= (w + mw):
		size.append([0, 0, W, H])
		return size

	def fc(t, mt, T):
		tmin, tmax = [], []
		t1 = mt
		while True:
			t0 = t1 - mt
			t1 = t0 + t
			if t1 >= T - mt:
				if t1 <= T:
					t1 = T
				else:
					t1 = T
					t0 = T - t
				tmin.append(t0)
				tmax.append(t1)
				break
			tmin.append(t0)
			tmax.append(t1)
		return tmin, tmax

	xmin, xmax = fc(w, mw, W)
	ymin, ymax = fc(h, mh, H)
	for y0, y1 in zip(ymin, ymax):
		for x0, x1 in zip(xmin, xmax):
			size.append([x0, y0, x1, y1])
	size.append([0, 0, W, H])
	return size
