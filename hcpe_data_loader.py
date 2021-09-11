import datetime
import glob
import itertools
import logging
import numpy as np
import tensorflow as tf
import cshogi
import hcpe_preprocessing


def load_hcpe(hcpe_num):

	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	board_data = {}
	y = []
	for hcpe_name in itertools.islice(glob.glob("./hcpe/**/*.hcpe"), hcpe_num):
		logging.info(hcpe_name)
		hcpes = np.fromfile(hcpe_name, dtype=cshogi.HuffmanCodedPosAndEval)
		board = cshogi.Board()
		for key in hcpe_preprocessing.cnn_board_to_features(board).keys():
			board_data.setdefault(key, [])
		for item in hcpes:
			board.set_hcp(item['hcp'])
			feature = hcpe_preprocessing.cnn_board_to_features(board)
			for key in feature.keys():
				board_data[key].append(feature[key])
			y.append(item['eval'])

	#board_data = {}
	for key in feature.keys():
		board_data[key] = np.array(board_data[key], dtype=float)
	board_data['y'] = np.array(y, dtype=float)

	return tf.data.Dataset.from_tensor_slices(board_data)
