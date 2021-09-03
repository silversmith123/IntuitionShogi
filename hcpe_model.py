import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate
from tensorflow.keras import Sequential, Model

def weight_variable(list):
    initial = tf.constant(list)
    return tf.Variable(initial)

class basic_multiply_layer(tf.keras.layers.Layer):
	def __init__(self, units=32, input_dim=32, name=''):
		super(basic_multiply_layer, self).__init__(name=name)
		self.w = weight_variable([0.0, 100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 2000.0, 600.0, 650.0, 700.0, 1000.0, 1100.0, 1200.0, 0.0, 0.0, -100.0, -300.0, -400.0, -500.0, -600.0, -800.0, -1000.0, -2000.0, -600.0, -650.0, -700.0, -1000.0, -1100.0, -1200.0])

	def call(self, inputs):
		return tf.multiply(inputs,self.w)


class motikoma_multiply_layer(tf.keras.layers.Layer):
	def __init__(self, units=32, input_dim=32, name=''):
		super(motikoma_multiply_layer, self).__init__(name=name)
		self.w = weight_variable([100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0])

	def call(self, inputs):
		return tf.multiply(inputs,self.w)

class reduce_sum_layer(tf.keras.layers.Layer):
	def __init__(self, units=32, input_dim=32, name=''):
		super(reduce_sum_layer, self).__init__(name=name)
	def call(self, inputs):
		return tf.reduce_sum(inputs, axis=-1)

class math_log1p_layer(tf.keras.layers.Layer):
	def __init__(self, units=32, input_dim=32, name=''):
		super(math_log1p_layer, self).__init__(name=name)
	def call(self, inputs):
		return tf.math.log1p(inputs)

def CNNModel():
	koma_board_inputs = tf.keras.Input(shape=(9, 9, 31), name='koma_board')
	hand_koma_b_inputs = tf.keras.Input(shape=(7,), name='hand_koma_b')
	hand_koma_w_inputs = tf.keras.Input(shape=(7,), name='hand_koma_w')
	inputs = {}
	inputs['koma_board'] = koma_board_inputs
	inputs['hand_koma_b'] = hand_koma_b_inputs
	inputs['hand_koma_w'] = hand_koma_w_inputs

	koma_value = basic_multiply_layer(name='koma_value_multiply')(koma_board_inputs)
	koma_value = reduce_sum_layer(name='koma_value_reduce_1')(koma_value)
	koma_value = reduce_sum_layer(name='koma_value_reduce_2')(koma_value)
	koma_value = reduce_sum_layer(name='koma_value_reduce_3')(koma_value)

	hand_value_b = motikoma_multiply_layer(name='hand_value_b1')(hand_koma_b_inputs)
	hand_value_b = reduce_sum_layer(name='hand_value_b2')(hand_value_b)

	hand_value_w = motikoma_multiply_layer(name='hand_value_w1')(hand_koma_w_inputs)
	hand_value_w = reduce_sum_layer(name='hand_value_w2')(hand_value_w)

	koma_value = tf.keras.layers.add([koma_value, hand_value_b], name='koma_value_add')
	koma_value = tf.keras.layers.subtract([koma_value, hand_value_w], name='koma_value_output')

	conv = tf.keras.layers.Conv2D(
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu,
		input_shape=(9, 9, 31),
		data_format='channels_last',
		name='koma_board_conv1'
	)(koma_board_inputs)
	pool = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=3, name='koma_board_pool1')(conv)
	conv2 = tf.keras.layers.Conv2D(
		filters=32,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu,
		data_format='channels_last',
		name='koma_board_conv2'
	)(pool)
	pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=3, name='koma_board_pool2')(conv2)
	flatten = tf.keras.layers.Flatten(name='koma_board_flatten')(pool2)
	dense = tf.keras.layers.Dense(units=10, activation=tf.nn.relu, name='koma_board_dence1')(flatten)
	dense2 = tf.keras.layers.Dense(units=1, activation=tf.nn.softmax, name='koma_board_dence2')(dense)
	koma_board_value = math_log1p_layer(name='koma_board_log1p')(dense2)
	koma_board_value = reduce_sum_layer(name='koma_board_output')(koma_board_value)

	outputs = tf.keras.layers.add([koma_board_value, koma_value], name='outputs')

	return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='functional')
