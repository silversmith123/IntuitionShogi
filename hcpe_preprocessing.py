import numpy as np
import cshogi

def cnn_board_to_features(board):

	# 駒の配列
	koma = np.array(board.pieces, dtype=int)
	koma_board = np.identity(31, dtype = float)[koma]
	koma_board = koma_board.reshape(9, 9, 31)
	hand_koma_b = board.pieces_in_hand[cshogi.BLACK]
	hand_koma_w = board.pieces_in_hand[cshogi.WHITE]

	features = {}
	features['koma_board'] = koma_board
	features['hand_koma_b'] = hand_koma_b
	features['hand_koma_w'] = hand_koma_w

	return features
