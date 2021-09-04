import numpy as np
import cshogi

def cnn_board_to_features(board):

	# 駒の配列 手番の側を先手とする
	koma = np.array(board.pieces, dtype=int)
	koma_board = np.identity(31, dtype = float)[koma]
	if board.turn == cshogi.WHITE:
		koma_board = koma_board[::, [0,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 ,28, 29, 30, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
		koma_board = koma_board[::-1, ::]
	koma_board = koma_board.reshape(9, 9, 31)

	if board.turn == cshogi.BLACK:
		hand_koma_b = board.pieces_in_hand[cshogi.BLACK]
		hand_koma_w = board.pieces_in_hand[cshogi.WHITE]
	else:
		hand_koma_b = board.pieces_in_hand[cshogi.WHITE]
		hand_koma_w = board.pieces_in_hand[cshogi.BLACK]

	features = {}
	features['koma_board'] = koma_board
	features['hand_koma_b'] = hand_koma_b
	features['hand_koma_w'] = hand_koma_w

	return features
