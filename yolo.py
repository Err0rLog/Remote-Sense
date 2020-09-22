import os

import keras.backend as K
from keras.layers import Input
from keras.utils import multi_gpu_model
from yolov3.model import yolo_eval, yolo_body
from yolov3.utils import letterbox_image, wh2xy, draw_box, nms, segmentation
from timeit import default_timer as timer
import numpy as np


class YOLO:
	def __init__(self):
		# 设置默认属性
		self.weight_path = 'model_data/weight.h5'
		self.anchors_path = 'model_data/anchors.txt'
		self.classes_path = 'model_data/classes.txt'  # English only
		self.model_image_size = (576, 576)  # hw
		# cv is GBR
		self.colors = [(0, 191, 255), (127, 255, 212), (238, 130, 238)]
		self.class_names = self._get_class()
		self.anchors = self._get_anchors()
		self.sess = K.get_session()
		self.score = 0.3
		self.iou = 0.45
		self.gpu_num = 1
		self.yolo_model = None
		self.input_image_shape = None
		# 3个运算结果 tensor
		self.boxes, self.scores, self.classes = self.generate()

	def _get_class(self):
		# expanduser: 把path中包含的"~"和"~user"转换成用户目录
		classes_path = os.path.expanduser(self.classes_path)
		with open(classes_path, 'r') as f:
			class_names = f.readlines()
		class_names = [s.strip() for s in class_names]
		return class_names

	def _get_anchors(self):
		anchors_path = os.path.expanduser(self.anchors_path)
		with open(anchors_path) as f:
			anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		return np.array(anchors).reshape(-1, 2)

	def generate(self):
		"""未运算, 返回model结果的tensor"""
		weight_path = os.path.expanduser(self.weight_path)
		assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

		# Load model, or construct model and load weights.
		num_anchors = len(self.anchors)
		num_classes = len(self.class_names)
		h, w = self.model_image_size
		assert os.path.exists(self.weight_path), 'weight文件不存在'
		self.yolo_model = yolo_body(Input(shape = (h, w, 3)), num_anchors // 3, num_classes)
		self.yolo_model.load_weights(self.weight_path)  # make sure model, anchors and classes match

		print('{} model, anchors, and classes loaded.'.format(weight_path))

		# Generate output tensor targets for filtered bounding boxes.
		# placeholder: 实例化一个占位符张量并返回它
		# input_image_shape 只输入wh
		self.input_image_shape = K.placeholder(shape = (2,))
		if self.gpu_num >= 2:
			self.yolo_model = multi_gpu_model(self.yolo_model, gpus = self.gpu_num)
		boxes, scores, classes = yolo_eval(self.yolo_model.output,
		                                   self.anchors,
		                                   len(self.class_names),
		                                   self.input_image_shape,
		                                   score_threshold = self.score)
		# boxes: xywh
		return boxes, scores, classes

	def detect_image(self, image: np.ndarray):
		"""检测图像"""
		start = timer()
		image_h, image_w = image.shape[0:2]

		assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
		assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
		# reversed: 反向迭代器, 默认输入为hw, 要转化为wh
		boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
		image_data = boxed_image.astype('float32')

		print(image_data.shape)
		image_data /= 255.
		image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

		# run run run
		out_boxes, out_scores, out_classes = self.sess.run(
			[self.boxes, self.scores, self.classes],
			feed_dict = {
				self.yolo_model.input: image_data,  # 替换图中的某个tensor的值
				self.input_image_shape: [image_w, image_h],
				# learning_phase, 学习阶段标志是一个布尔张量（0 = test，1 = train)
				K.learning_phase(): 0
				})
		print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

		out_boxes = wh2xy(out_boxes)
		keep = nms(out_boxes, out_scores, self.iou)  # box中为角度
		out_boxes = out_boxes[keep]  # [N, 5]
		out_scores = out_scores[keep]  # [N,]
		out_classes = out_classes[keep]  # [N,]

		print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

		out_boxes = np.floor(out_boxes).astype(np.int)

		# draw
		image = draw_box(image, out_boxes, out_scores, out_classes, self.colors, self.class_names)
		end = timer()

		print('time: ', end - start)
		return image

	def detect_big_image(self, image: np.ndarray):
		"""检测大图像"""
		start = timer()
		assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
		assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
		# reversed: 反向迭代器, 默认输入为hw, 要转化为wh
		size = segmentation(image, self.model_image_size)
		H, W, _ = image.shape
		all_box, all_score, all_classes = [], [], []
		print(image.shape)
		for t in size:
			img = image[t[1]:t[3], t[0]:t[2]]
			boxed_image = letterbox_image(img, tuple(reversed(self.model_image_size)))
			image_data = boxed_image.astype('float32')
			image_h, image_w, _ = image_data.shape

			image_data /= 255.0
			image_data = np.expand_dims(image_data, 0)

			out_boxes, out_scores, out_classes = self.sess.run(
				[self.boxes, self.scores, self.classes],
				feed_dict = {
					self.yolo_model.input: image_data,  # 替换图中的某个tensor的值
					self.input_image_shape: [image_w, image_h],
					# learning_phase, 学习阶段标志是一个布尔张量（0 = test，1 = train)
					K.learning_phase(): 0
					})
			out_boxes[..., 0] += t[0]
			out_boxes[..., 1] += t[1]
			all_box.append(out_boxes)
			all_score.append(out_scores)
			all_classes.append(out_classes)

		out_boxes = np.concatenate(all_box)
		out_scores = np.concatenate(all_score)
		out_classes = np.concatenate(all_classes)
		out_boxes = wh2xy(out_boxes)

		keep = nms(out_boxes, out_scores, self.iou)
		print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
		out_boxes = np.floor(out_boxes[keep]).astype(np.int)  # [N, 5]
		out_scores = out_scores[keep]  # [N,]
		out_classes = out_classes[keep]  # [N,]

		out_boxes[..., 0:2][out_boxes[..., 0:2] < 0] = 0
		out_boxes[..., 2:3][out_boxes[..., 2:3] > (W - 1)] = W - 1
		out_boxes[..., 3:4][out_boxes[..., 3:4] > (H - 1)] = H - 1

		image = draw_box(image, out_boxes, out_scores, out_classes, self.colors, self.class_names)
		end = timer()

		print('time: ', end - start)
		return image

	def close_session(self):
		self.sess.close()
