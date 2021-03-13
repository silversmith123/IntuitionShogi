import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate
from tensorflow.keras import Sequential, Model
import shogi

def sfen_to_features(sfen):
	board = shogi.Board(sfen=sfen)
	return board_to_features(board)

def board_to_features(board):
    # 駒の配列
    koma_b = [board.piece_at(sq).piece_type if board.piece_at(
        sq) != None and board.piece_at(sq).color == shogi.BLACK else 0 for sq in range(81)]
    koma_w = [board.piece_at(sq).piece_type if board.piece_at(
        sq) != None and board.piece_at(sq).color == shogi.WHITE else 0 for sq in range(81)]
    koma_b = np.array(koma_b)
    koma_b = np.eye(15)[koma_b.reshape(-1)]
    koma_w = np.array(koma_w)
    koma_w = np.eye(15)[koma_w.reshape(-1)]
    # 攻撃している駒の配列
    atk_koma_b = np.zeros((81, 15), dtype=int)
    atk_koma_w = np.zeros((81, 15), dtype=int)
    difence_koma_b = np.zeros((81, 15), dtype=int)
    difence_koma_w = np.zeros((81, 15), dtype=int)

    for sq in range(81):
        for atk_sq in board.attackers(shogi.BLACK, sq):
            if board.piece_at(atk_sq):
                atk_koma_b[sq][board.piece_at(atk_sq).piece_type] = 1
            if board.piece_at(atk_sq):
                difence_koma_b[sq][board.piece_at(atk_sq).piece_type] = 1
        for atk_sq in board.attackers(shogi.WHITE, sq):
            if board.piece_at(atk_sq):
                atk_koma_w[sq][board.piece_at(atk_sq).piece_type] = 1
            if board.piece_at(atk_sq):
                difence_koma_w[sq][board.piece_at(atk_sq).piece_type] = 1

        # 持ち駒
    hand_koma_b = [0] * 7
    p = board.pieces_in_hand[shogi.BLACK]
    for piece_type in shogi.PIECE_TYPES[0:7]:
        hand_koma_b[shogi.PIECE_TYPES.index(piece_type)] = p[piece_type]
    hand_koma_w = [0] * 7
    p = board.pieces_in_hand[shogi.WHITE]
    for piece_type in shogi.PIECE_TYPES[0:7]:
        hand_koma_w[shogi.PIECE_TYPES.index(piece_type)] = p[piece_type]

        # 全て先手番から見た位置に変換
    if board.turn == shogi.WHITE:
        # 持ち駒は交換するだけ
        hand_koma_b, hand_koma_w = hand_koma_w, hand_koma_b
        # 盤上の駒は交換後、位置をひっくり返す
        koma_b, koma_w = koma_w, koma_b
        koma_b = np.array([koma_b[shogi.SQUARES_L90(shogi.SQUARES_L90[sq])] for sq in range(81)])
        koma_w = np.array([koma_w[shogi.SQUARES_L90(shogi.SQUARES_L90[sq])] for sq in range(81)])
        atk_koma_b, atk_koma_w = atk_koma_w, atk_koma_b
        atk_koma_b = np.array([atk_koma_b[shogi.SQUARES_L90(shogi.SQUARES_L90[sq])]
                               for sq in range(81)])
        atk_koma_w = np.array([atk_koma_w[shogi.SQUARES_L90(shogi.SQUARES_L90[sq])]
                               for sq in range(81)])
    # 9*9にrehshape
    #koma_b = koma_b.reshape(9, 9, 15)
    #koma_w = koma_w.reshape(9, 9, 15)
    #atk_koma_b = atk_koma_b.reshape(9, 9, 15)
    #atk_koma_w = atk_koma_w.reshape(9, 9, 15)
    # features['turn'].append(board.turn)
    features = {}
    features['koma_b'] = koma_b
    features['koma_w'] = koma_w
    features['atk_koma_b'] = atk_koma_b
    features['atk_koma_w'] = atk_koma_w
    features['hand_koma_b'] = hand_koma_b
    features['hand_koma_w'] = hand_koma_w
    features['difence_koma_b'] = difence_koma_b
    features['difence_koma_w'] = difence_koma_w
    # features['eval'].append(eval)
    # features['gameResult'].append(gameResult)
    # features['turn']=np.array(features['turn'])
    features['koma_b'] = np.array(features['koma_b'])
    features['koma_w'] = np.array(features['koma_w'])
    features['atk_koma_b'] = np.array(features['atk_koma_b'])
    features['atk_koma_w'] = np.array(features['atk_koma_w'])
    features['difence_koma_b'] = np.array(features['difence_koma_b'])
    features['difence_koma_w'] = np.array(features['difence_koma_w'])
    features['hand_koma_b'] = np.array(features['hand_koma_b'])
    features['hand_koma_w'] = np.array(features['hand_koma_w'])
    # features['eval']=np.array(features['eval'])
    # features['gameResult']=np.array(features['gameResult'])

    return features


def eval_sfen(sfen):
    features = sfen_to_features(sfen)
    eval = evaluationFun(features)

    return eval


def weight_variable(list):
    initial = tf.constant(list)
    return tf.Variable(initial)


def basic_value():
    W = weight_variable([100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0,
                         2000.0, -100.0, -300.0, -400.0, -500.0, -600.0, -800.0, -1000.0])
    return W


class basic_multiply_layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(basic_multiply_layer, self).__init__()
        self.w = basic_value()

    def call(self, inputs):
        return tf.multiply(inputs,self.w)


def motikoma_value():

    W = weight_variable([100.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0])
    return W

class motikoma_multiply_layer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(motikoma_multiply_layer, self).__init__()
        self.w = motikoma_value()

    def call(self, inputs):
        return tf.multiply(inputs,self.w)


def bias_variable():
    initial = tf.constant(0.1)
    return tf.Variable(initial)

def create_model():

    # for key, value in features.items():
    #    features[key] = tf.cast(value, tf.float32)
	# koma_value_b = tf.multiply(features['koma_b'], W_basic_value)
    # koma_value_w = tf.multiply(features['koma_w'], W_basic_value)
    # atk_value_b = tf.multiply(features['atk_koma_b'], W_basic_value)
    # atk_value_w = tf.multiply(features['atk_koma_w'], W_basic_value)
    # difence_value_b = tf.multiply(features['difence_koma_b'], W_basic_value)
    # difence_value_w = tf.multiply(features['difence_koma_w'], W_basic_value)value
	with tf.name_scope("komavalue_Block"):
		multiply_layer = basic_multiply_layer()

		inputs_koma_b = Input(shape=(81, 15), name='inputs_koma_b')		
		multiply_b = multiply_layer(inputs_koma_b)
		koma_value_b=tf.reduce_sum(multiply_b, axis=[1, 2])

		inputs_koma_w = Input(shape=(81, 15), name='inputs_koma_w')
		multiply_w = multiply_layer(inputs_koma_w)
		koma_value_w=tf.reduce_sum(multiply_w, axis=[1, 2])

		#return koma_value_model_b

	
	with tf.name_scope("handkomavalue_Block"):
		motikoma_layer = motikoma_multiply_layer()

		inputs_hand_koma_b = Input(shape=(7), name='inputs_hand_koma_b')		
		multiply_hand_b = motikoma_layer(inputs_hand_koma_b)
		hand_value_b=tf.reduce_sum(multiply_hand_b, axis=1)

		inputs_hand_koma_w = Input(shape=(7), name='inputs_hand_koma_w')
		multiply_hand_w = motikoma_layer(inputs_hand_koma_w)
		hand_value_w=tf.reduce_sum(multiply_hand_w, axis=1)

	with tf.name_scope("atkvalue_Block"):
		inputs_atk_koma_b = Input(shape=(81, 15), name='inputs_atk_koma_b')
		inputs_atk_koma_w = Input(shape=(81, 15), name='inputs_atk_koma_w')
		atk_multiply_b = multiply_layer(inputs_atk_koma_b)
		atk_multiply_b = tf.reshape(atk_multiply_b, [-1, 9, 9, 15])
		atk_multiply_w = multiply_layer(inputs_atk_koma_w)
		atk_multiply_w = tf.reshape(atk_multiply_w, [-1, 9, 9, 15])

		atk_value_max_b = tf.reduce_max(atk_multiply_b, axis=-1)
		atk_value_total_b = tf.reduce_sum(atk_multiply_b, axis=-1)
		atk_value_max_w = tf.reduce_max(atk_multiply_w, axis=-1)
		atk_value_total_w = tf.reduce_sum(atk_multiply_w, axis=-1)

		#atk_value = Concatenate(axis=1)()
		atk_value=tf.stack([atk_value_max_b, atk_value_total_b, atk_value_max_w, atk_value_total_w], axis=-1)

		atk_conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu,
			data_format='channels_last'
		)(atk_value)
		# プーリング層作成
		pool = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=3)(atk_conv)
		pool_flat = tf.reshape(pool, [-1, 3 * 3 * 64])
		dense = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(pool_flat)
		dense = tf.reduce_sum(dense, axis=-1)

	with tf.name_scope("merge_Block"):
		merged = tf.stack([koma_value_b, koma_value_w, hand_value_b, hand_value_w, dense], axis=-1)
		#merged = [tf.constant([1.0, 0.0, 0.0, 0.0, 0.0])]
		W_total = tf.constant([1.0, -1.0, 1.0, -1.0, 1.0])
		merged = tf.tensordot(merged, W_total, axes=[[1],[0]])
		#model = Model([inputs_koma_b, inputs_koma_w, inputs_hand_koma_b, inputs_hand_koma_w, inputs_atk_koma_b, inputs_atk_koma_w], merged)
		model = Model([inputs_koma_b, inputs_koma_w, inputs_hand_koma_b, inputs_hand_koma_w, inputs_atk_koma_b, inputs_atk_koma_w], merged)
	
	return model


def evaluationFun(features):
	model = create_model()
	model.compile(optimizer='adam', loss='mean_absolute_error')
	inputs_koma_b = np.array([features['koma_b']])
	inputs_koma_w = np.array([features['koma_w']])
	inputs_hand_koma_b = np.array([features['hand_koma_b']])
	inputs_hand_koma_w = np.array([features['hand_koma_w']])
	inputs_atk_koma_b = np.array([features['atk_koma_b']])
	inputs_atk_koma_w = np.array([features['atk_koma_w']])
	inputs = [inputs_koma_b, inputs_koma_w, inputs_hand_koma_b, inputs_hand_koma_w, inputs_atk_koma_b, inputs_atk_koma_w]
	print(model.predict({'inputs_koma_b': inputs_koma_b, 'inputs_koma_w': inputs_koma_w, 'inputs_hand_koma_b':inputs_hand_koma_b, 'inputs_hand_koma_w': inputs_hand_koma_w, 'inputs_atk_koma_b': inputs_atk_koma_b, 'inputs_atk_koma_w': inputs_atk_koma_w}, verbose= 2))
	#model.summary()


@profile
def evaluate(board, model):
	features = board_to_features(board)
	inputs_koma_b = np.array([features['koma_b']])
	inputs_koma_w = np.array([features['koma_w']])
	inputs_hand_koma_b = np.array([features['hand_koma_b']])
	inputs_hand_koma_w = np.array([features['hand_koma_w']])
	inputs_atk_koma_b = np.array([features['atk_koma_b']])
	inputs_atk_koma_w = np.array([features['atk_koma_w']])
	inputs = [inputs_koma_b, inputs_koma_w, inputs_hand_koma_b, inputs_hand_koma_w, inputs_atk_koma_b, inputs_atk_koma_w]
	eval = model.predict({'inputs_koma_b': inputs_koma_b, 'inputs_koma_w': inputs_koma_w, 'inputs_hand_koma_b':inputs_hand_koma_b, 'inputs_hand_koma_w': inputs_hand_koma_w, 'inputs_atk_koma_b': inputs_atk_koma_b, 'inputs_atk_koma_w': inputs_atk_koma_w}, verbose= False)[0]
	if board.turn == shogi.WHITE:
		eval = -eval
	return eval

def init_model():
	model = create_model()
	model.compile(optimizer='adam', loss='mean_absolute_error')
	return model

