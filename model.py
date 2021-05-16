import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate
from tensorflow.keras import Sequential, Model
import shogi

def weight_variable(list):
    initial = tf.constant(list)
    return tf.Variable(initial)


class basic_multiply_layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(basic_multiply_layer, self).__init__()
        self.w = weight_variable([0.0, 100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
                         2000.0, 600.0, 650.0, 700.0, 1000.0, 1100.0, 1200])

    def call(self, inputs):
        return tf.multiply(inputs,self.w)


class motikoma_multiply_layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(motikoma_multiply_layer, self).__init__()
        self.w = weight_variable([100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0])

    def call(self, inputs):
        return tf.multiply(inputs,self.w)


def bias_variable():
    initial = tf.constant(0.1)
    return tf.Variable(initial)
class IntuitionModel(Model):

	def __init__(self):
		self.debug = False
		super(IntuitionModel, self).__init__()
		self.basic_multiply_layer = basic_multiply_layer()
		self.motikoma_multiply_layer = motikoma_multiply_layer()
		self.pool = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=3)
		self.dense = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
		self.atk_conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu,
			data_format='channels_last'
		)

	@tf.function
	def call(self, inputs):
		inputs_koma_b = inputs['inputs_koma_b']
		multiply_b = self.basic_multiply_layer(inputs_koma_b)
		koma_value_b=tf.reduce_sum(multiply_b, axis=[1, 2])
		
		inputs_koma_w = inputs['inputs_koma_w']
		multiply_w = self.basic_multiply_layer(inputs_koma_w)
		koma_value_w = tf.reduce_sum(multiply_w, axis=[1, 2])

		inputs_hand_koma_b = inputs['inputs_hand_koma_b']	
		multiply_hand_b = self.motikoma_multiply_layer(inputs_hand_koma_b)
		hand_value_b = tf.reduce_sum(multiply_hand_b, axis=1)

		inputs_hand_koma_w = inputs['inputs_hand_koma_w']	
		multiply_hand_w = self.motikoma_multiply_layer(inputs_hand_koma_w)
		hand_value_w = tf.reduce_sum(multiply_hand_w, axis=1)

		inputs_atk_koma_b = inputs['inputs_atk_koma_b']
		inputs_atk_koma_w = inputs['inputs_atk_koma_w']
		atk_multiply_b = self.basic_multiply_layer(inputs_atk_koma_b)
		atk_multiply_b = tf.reshape(atk_multiply_b, [-1, 9, 9, 15])
		atk_multiply_w = self.basic_multiply_layer(inputs_atk_koma_w)
		atk_multiply_w = tf.reshape(atk_multiply_w, [-1, 9, 9, 15])
		atk_num_b = tf.reshape(inputs_atk_koma_b, [-1, 9, 9, 15])
		atk_num_w = tf.reshape(inputs_atk_koma_b, [-1, 9, 9, 15])

		atk_value_max_b = tf.reduce_max(atk_multiply_b, axis=-1)
		atk_value_max_w = tf.reduce_max(atk_multiply_w, axis=-1)
		atk_num_b = tf.reduce_max(atk_num_b, axis=-1)
		atk_num_w = tf.reduce_max(atk_num_w, axis=-1)

		difence_multiply_b = self.basic_multiply_layer(inputs_koma_b)
		difence_multiply_b = tf.reshape(difence_multiply_b, [-1, 9, 9, 15])
		difence_multiply_w = self.basic_multiply_layer(inputs_koma_w)
		difence_multiply_w = tf.reshape(difence_multiply_w, [-1, 9, 9, 15])

		difence_value_b = tf.reduce_max(difence_multiply_b, axis=-1)
		difence_value_w = tf.reduce_max(difence_multiply_w, axis=-1)


		if self.debug:
			print('koma_value_b')
			print(koma_value_b)
			print('koma_value_w')
			print(koma_value_w)
			print('hand_value_b')
			print(hand_value_b)
			print('hand_value_w')
			print(hand_value_w)
			print('atk_value_max_b')
			print(atk_value_max_b)
			print('atk_num_b')
			print(atk_num_b)
			print('atk_value_max_w')
			print(atk_value_max_w)
			print('atk_num_w')
			print(atk_num_w)
			print('difence_value_b')
			print(difence_value_b)
			print('difence_value_w')
			print(difence_value_w)

		atk_value=tf.stack([atk_value_max_b, atk_num_b, atk_value_max_w, atk_num_w, difence_value_b, difence_value_w], axis=-1)

		atk_conv = self.atk_conv(atk_value)
		# プーリング層作成
		pool = self.pool(atk_conv)
		pool_flat = tf.reshape(pool, [-1, 3 * 3 * 64])
		dense = self.dense(pool_flat)
		dense = tf.reduce_sum(dense, axis=-1)

		merged = tf.stack([koma_value_b, koma_value_w, hand_value_b, hand_value_w, dense], axis=-1)
		#merged = [tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])]
		W_total = tf.constant([1.0, -1.0, 1.0, -1.0, 1.0])
		merged = tf.tensordot(merged, W_total, axes=[[1],[0]])

		return merged

	def train_step(self, data):
		target_features = []
		best_feature = []
		with tf.GradientTape() as tape:
			y_pred = []

			tgt = {}
			tgt['inputs_koma_b'] = data['koma_b']
			tgt['inputs_koma_w'] = data['koma_w']
			tgt['inputs_atk_koma_b'] = data['atk_koma_b']
			tgt['inputs_atk_koma_w'] = data['atk_koma_w']
			tgt['inputs_hand_koma_b'] = data['hand_koma_b']
			tgt['inputs_hand_koma_w'] = data['hand_koma_w']
			tgt['inputs_difence_koma_b'] = data['difence_koma_b']
			tgt['inputs_difence_koma_w'] = data['difence_koma_w']

			y_pred = self(tgt, training=True)

			y_tgt = {}
			y_tgt['inputs_koma_b'] = tf.stack([data['y_koma_b'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_koma_w'] = tf.stack([data['y_koma_w'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_atk_koma_b'] = tf.stack([data['y_atk_koma_b'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_atk_koma_w'] = tf.stack([data['y_atk_koma_w'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_hand_koma_b'] = tf.stack([data['y_hand_koma_b'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_hand_koma_w'] = tf.stack([data['y_hand_koma_w'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_difence_koma_b'] = tf.stack([data['y_difence_koma_b'] for i in range(data['koma_b'].shape[0])])
			y_tgt['inputs_difence_koma_w'] = tf.stack([data['y_difence_koma_w'] for i in range(data['koma_b'].shape[0])])

			y = self(y_tgt, training=True)

			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

			trainable_vars = self.trainable_variables
			gradients = tape.gradient(loss, trainable_vars)
			self.optimizer.apply_gradients(zip(gradients, trainable_vars))
			return {m.name: m.result() for m in self.metrics}
