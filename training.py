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


def learn():
	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	original_model = model.IntuitionModel()
	original_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	board_features = []
	best_features = []
	for csa_name in itertools.islice(glob.glob("./kif/*/*.csa"), 100):
		logging.info(csa_name)
		kif = CSA.Parser().parse_str(open(csa_name).read())
		board = shogi.Board(kif[0]['sfen'])
		prev_features = preprocessing.board_to_features(board)
		for best_move in kif[0]['moves']:
			for cand_move in shogi.LegalMoveGenerator(board):
				move_features = []
				if cand_move.usi() == best_move:
					continue
				move_features.append(preprocessing.update_features(board, cand_move, prev_features))
				board.pop()
			board.push(shogi.Move.from_usi(best_move))
			if len(move_features) == 0:
				continue
			board_features.append(move_features)
			prev_features = preprocessing.board_to_features(board)
			best_features.append(prev_features)

	learn_set = preprocessing.ds_from_features(board_features, best_features)

	original_model.fit(learn_set, epochs=1, callbacks=[tensorboard_callback])
	original_model.save('./model/intuitionshogi.pb')

learn()
