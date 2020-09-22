# -*- coding: utf-8 -*-
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolov3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolov3.utils import get_random_data


def train():
	train_path = 'train2.txt'
	weights_path = 'darknet53_weights.h5'
	log_dir = 'logs/p002/'
	classes_path = 'model_data/classes.txt'
	anchors_path = 'model_data/anchors.txt'
	class_names = get_classes(classes_path)
	num_classes = len(class_names)
	anchors = get_anchors(anchors_path)
	input_shape = (416, 416)
	batch_size1, batch_size2 = 2, 8

	model = create_model(input_shape, anchors, num_classes,
	                     freeze_body = 0, weights_path = weights_path, load_pretrained = False)
	# model.save('yolo.h5')
	# Tensorboard 基本可视化
	logging = TensorBoard(log_dir = log_dir)
	# 在每个训练期之后保存模型
	checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
	                             monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 3)
	# 当标准评估停止提升时，降低学习速率。
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1)
	# 当被监测的数量不再提升，则停止训练。修改以提早结束
	early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 50, verbose = 1)

	val_split = 0.1
	with open(train_path, 'r') as f:
		lines = f.readlines()
	np.random.seed(10101)
	np.random.shuffle(lines)
	np.random.seed(None)
	# 测试数据
	num_val = int(len(lines) * val_split)  # 0.1 * len(lines)
	# 训练数据
	num_train = len(lines) - num_val  # 0.9 * len(lines)

	# 第一次训练, 先冻结部分权重------------------------------------
	# 这里model的loss设置为model的输出,因为model是yolo_body和yolo_loss的间接了的整体
	model.compile(optimizer = Adam(lr = 1e-3), loss = {
		# use custom yolo_loss Lambda layer.
		'yolo_loss': lambda y_true, y_pred: y_pred})

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size1))
	# 使用 Python 生成器（或 Sequence 实例）逐批生成的数据，按批次训练模型

	model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size1, input_shape, anchors, num_classes),
	                    steps_per_epoch = max(1, num_train // batch_size1),
	                    verbose = 1,
	                    validation_data = data_generator_wrapper(lines[num_train:], batch_size1, input_shape, anchors,
	                                                             num_classes),
	                    validation_steps = max(1, num_val // batch_size1),
	                    epochs = 5,
	                    initial_epoch = 0,
	                    callbacks = [logging, checkpoint])
	# model.save_weights(log_dir + 'trained_weights_stage_2.h5')

	# # 第二次训练, 训练所有参数----------------------------------------
	for i in range(len(model.layers)):
		model.layers[i].trainable = True

	model.compile(optimizer = Adam(lr = 1e-4),  # 10e-7
	              loss = {'yolo_loss': lambda y_true, y_pred: y_pred})
	print('Unfreeze all of the layers.')

	# 吃GPU memory了
	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size2))
	model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size2, input_shape, anchors, num_classes),
	                    steps_per_epoch = max(1, num_train // batch_size2),
	                    verbose = 1,
	                    validation_data = data_generator_wrapper(lines[num_train:], batch_size2, input_shape,
	                                                             anchors, num_classes),
	                    validation_steps = max(1, num_val // batch_size2),
	                    epochs = 2,
	                    initial_epoch = 1,
	                    callbacks = [logging, checkpoint, reduce_lr, early_stopping])
	# model.save_weights(log_dir + 'weight.h5')  # logs/p001/weight.h5, 拷贝到model_data/weight.h5即可预测
	print('ok')


def get_classes(classes_path: str):
	with open(classes_path, 'r') as f:
		classes = f.readlines()
	classes = tuple(i.strip() for i in classes)
	return classes


def get_anchors(anchors_path):
	with open(anchors_path, 'r') as f:
		anchors = f.readline()
	anchors = [float(x) for x in anchors.split(',')]
	return np.array(anchors).reshape((-1, 2))


def create_model(input_shape, anchors, num_class, weights_path, freeze_body = 0, load_pretrained = False):
	"""
	创建一个model
	:param freeze_body: 冻结部分参数
	:param input_shape: 输入尺寸, hw
	:param anchors: 所有的anchors
	:param num_class: class数量
	:param weights_path: 已训练的权重的数量
	:param load_pretrained: 是否加载权重
	:return:
	"""
	K.clear_session()
	image_input = Input(shape = (416, 416, 3))
	h, w = input_shape
	num_anchors = len(anchors)
	out_dict = {0: 32, 1: 16, 2: 8}

	# y_true的输出
	y_true = [Input(shape = (h // out_dict[m], w // out_dict[m],
	                         num_anchors // 3, num_class + 5)) for m in range(3)]
	model_body = yolo_body(image_input, num_anchors // 3, num_class)

	# 加载已训练的weight
	if load_pretrained:
		import os
		assert os.path.exists(weights_path), 'weight文件不存在'
		model_body.load_weights(weights_path, by_name = True, skip_mismatch = True)
		print('Load weights {}.'.format(weights_path))
	if freeze_body in [1, 2]:
		# Freeze darknet53 body or freeze all but 3 output layers.
		num = (185, len(model_body.layers) - 3)[freeze_body - 1]
		for i in range(num):
			model_body.layers[i].trainable = False
		print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

	loss_argument = {'anchors': anchors,
	                 'num_classes': num_class,
	                 'ignore_thresh': 0.5}
	# yolo_loss封装为Layer对象(用来间接到yolo_body上, 使model的输出即为loss)
	# 因为这个model只用来训练
	model_loss = Lambda(yolo_loss, output_shape = (1,), name = 'yolo_loss',
	                    arguments = loss_argument)([*model_body.output, *y_true])

	model = Model([model_body.input, *y_true], model_loss)

	return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
	"""
	批量处理训练数据用以训练, 返回训练数据
	:param annotation_lines: train.txt读取的所有文本:
	:param batch_size:
	:param input_shape:
	:param anchors: 所有anchors
	:param num_classes:
	:return:
	"""
	n = len(annotation_lines)
	i = 0
	while True:
		image_data = []
		box_data = []
		for b in range(batch_size):
			if i == 0:
				np.random.shuffle(annotation_lines)
			image, box = get_random_data(annotation_lines[i], input_shape)
			image_data.append(image)
			box_data.append(box)
			i = (i + 1) % n
		image_data = np.array(image_data, dtype = np.float32)
		box_data = np.array(box_data, dtype = np.float32)
		y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
		# 返回(inputs, targets), 后者为标签,用于计算loss, 但loss已由model计算出来了, 故敷衍
		yield [image_data, *y_true], np.zeros(batch_size)
	# return [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
	n = len(annotation_lines)
	if n == 0 or batch_size <= 0:
		return None
	return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
	train()
