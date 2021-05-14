import datetime
import glob
import itertools
import logging
import numpy as np
import tensorflow as tf

from shogi import CSA
import shogi

import preprocessing
import model

def ds_from_board_features(board_features, best_features):
	koma_b = []
	koma_w = []
	atk_koma_b = []
	atk_koma_w = []
	hand_koma_b = []
	hand_koma_w = []
	difence_koma_b = []
	difence_koma_w = []

	y_koma_b = []
	y_koma_w = []
	y_atk_koma_b = []
	y_atk_koma_w = []
	y_hand_koma_b = []
	y_hand_koma_w = []
	y_difence_koma_b = []
	y_difence_koma_w = []

	for move_feature, best_feature in zip(board_features, best_features):
		record = {}
		record['koma_b'] = []
		record['koma_w'] = []
		record['atk_koma_b'] = []
		record['atk_koma_w'] = []
		record['hand_koma_b'] = []
		record['hand_koma_w'] = []
		record['difence_koma_b'] = []
		record['difence_koma_w'] = []

		for f in move_feature:
			record['koma_b'].append(f['koma_b'])
			record['koma_w'].append(f['koma_w'])
			record['atk_koma_b'].append(f['atk_koma_b'])
			record['atk_koma_w'].append(f['atk_koma_w'])
			record['hand_koma_b'].append(f['hand_koma_b'])
			record['hand_koma_w'].append(f['hand_koma_w'])
			record['difence_koma_b'].append(f['difence_koma_b'])
			record['difence_koma_w'].append(f['difence_koma_w'])

		record['y_koma_b'] = best_feature['koma_b']
		record['y_koma_w'] = best_feature['koma_w']
		record['y_atk_koma_b'] = best_feature['atk_koma_b']
		record['y_atk_koma_w'] = best_feature['atk_koma_w']
		record['y_hand_koma_b'] = best_feature['hand_koma_b']
		record['y_hand_koma_w'] = best_feature['hand_koma_w']
		record['y_difence_koma_b'] = best_feature['difence_koma_b']
		record['y_difence_koma_w'] = best_feature['difence_koma_w']
	
		koma_b.append(record['koma_b'])
		koma_w.append(record['koma_w'])
		atk_koma_b.append(record['atk_koma_b'])
		atk_koma_w.append(record['atk_koma_w'])
		hand_koma_b.append(record['hand_koma_b'])
		hand_koma_w.append(record['hand_koma_w'])
		difence_koma_b.append(record['difence_koma_b'])
		difence_koma_w.append(record['difence_koma_w'])
	
		y_koma_b.append(record['y_koma_b'])
		y_koma_w.append(record['y_koma_w'])
		y_atk_koma_b.append(record['y_atk_koma_b'])
		y_atk_koma_w.append(record['y_atk_koma_w'])
		y_hand_koma_b.append(record['y_hand_koma_b'])
		y_hand_koma_w.append(record['y_hand_koma_w'])
		y_difence_koma_b.append(record['y_difence_koma_b'])
		y_difence_koma_w.append(record['y_difence_koma_w'])

	koma_b = np.array(koma_b, dtype=float)
	koma_w = np.array(koma_w, dtype=float)
	atk_koma_b = np.array(atk_koma_b, dtype=float)
	atk_koma_w = np.array(atk_koma_w, dtype=float)
	hand_koma_b = np.array(hand_koma_b, dtype=float)
	hand_koma_w = np.array(hand_koma_w, dtype=float)
	difence_koma_b = np.array(difence_koma_b, dtype=float)
	difence_koma_w = np.array(difence_koma_w, dtype=float)

	y_koma_b = np.array(y_koma_b, dtype=float)
	y_koma_w = np.array(y_koma_w, dtype=float)
	y_atk_koma_b = np.array(y_atk_koma_b, dtype=float)
	y_atk_koma_w = np.array(y_atk_koma_w, dtype=float)
	y_hand_koma_b = np.array(y_hand_koma_b, dtype=float)
	y_hand_koma_w = np.array(y_hand_koma_w, dtype=float)
	y_difence_koma_b = np.array(y_difence_koma_b, dtype=float)
	y_difence_koma_w = np.array(y_difence_koma_w, dtype=float)


	dataset = tf.data.Dataset.from_tensor_slices(({'koma_b': koma_b, 'koma_w': koma_w, 'atk_koma_b': atk_koma_b, 'atk_koma_w': atk_koma_w, 'hand_koma_b': hand_koma_b, 'hand_koma_w': hand_koma_w, 'difence_koma_b': difence_koma_b, 'difence_koma_w': difence_koma_w, 'koma_b': koma_b, 'y_koma_b': y_koma_b, 'y_koma_w': y_koma_w, 'y_atk_koma_b': y_atk_koma_b, 'y_atk_koma_w': y_atk_koma_w, 'y_hand_koma_b': y_hand_koma_b, 'y_hand_koma_w': y_hand_koma_w, 'y_difence_koma_b': y_difence_koma_b, 'y_difence_koma_w': y_difence_koma_w}))
	return dataset


def learn():
	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	original_model = model.IntuitionModel()
	original_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	board_features = []
	best_features = []
	for csa_name in itertools.islice(glob.glob("./kif/*/*.csa"), 1):
		logging.info(csa_name)
		kif = CSA.Parser().parse_str(open(csa_name).read())
		board = shogi.Board(kif[0]['sfen'])
		for best_move in kif[0]['moves']:
			for cand_move in shogi.LegalMoveGenerator(board):
				move_features = []
				if cand_move.usi() == best_move:
					continue
				board.push(cand_move)
				move_features.append(preprocessing.board_to_features(board))
				board.pop()
			board.push(shogi.Move.from_usi(best_move))
			if len(move_features) == 0:
				continue
			board_features.append(move_features)
			best_features.append(preprocessing.board_to_features(board))

	learn_set = preprocessing.ds_from_features(board_features, best_features)

	original_model.fit(learn_set, epochs=1, callbacks=[tensorboard_callback])
	original_model.save('./model/intuitionshogi.pb')

learn()
