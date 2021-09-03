# -*- coding: utf-8 -*-
import cshogi
import hcpe_preprocessing


def show_feature(feature):
	sfen_rank = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
	def _print(feature):
		print('	'.join(['', '9', '8', '7', '6', '5', '4', '3', '2', '1']))
		for rank in range(9):
			line = sfen_rank[rank]
			for col in range(9):
				for piece in range(31):
					if not feature[(8-col)][rank][piece]:
						continue
					if piece < 15:
						val = ' ' + cshogi.PIECE_JAPANESE_SYMBOLS[piece]
					elif piece == 15 or piece == 16:
						val = " 　"
					if piece > 16:
						val = 'v' + cshogi.PIECE_JAPANESE_SYMBOLS[piece - 16]
				line += '	' +val
			print(line)

	for type in ['koma_board']:
		print(type)
		_print(feature[type])

def test_case1():
	print('初期局面')
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
	board = cshogi.Board()
	board.set_sfen(sfen)
	feature =hcpe_preprocessing.cnn_board_to_features(board)
	show_feature(feature)
	print('76歩')
	board.push_usi('7g7f')
	feature = hcpe_preprocessing.cnn_board_to_features(board)
	show_feature(feature)

test_case1()