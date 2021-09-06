import glob
import itertools
import logging
import numpy as np
from os.path import basename
import csv
import cshogi


def hcpe_to_csv(hcpe_num):

	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	board_data = {}
	y = []
	for hcpe_name in itertools.islice(glob.glob("./hcpe/**/*.hcpe"), hcpe_num):
		logging.info(hcpe_name)
		with open('analysis/csv/' + basename(hcpe_name)[:-4] + "csv", 'w', newline='') as csvfile:
			fieldnames = ['zobrist_hash', 'turn', 'sfen', 'CSAPos', 'eval']
			fieldnames.extend(cshogi.SQUARE_NAMES)
			fieldnames.extend(cshogi.HAND_PIECE_JAPANESE_SYMBOLS)
			fieldnames.extend(['v' + symbol for symbol  in  cshogi.HAND_PIECE_JAPANESE_SYMBOLS])
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator='\n')
			writer.writeheader()
			hcpes = np.fromfile(hcpe_name, dtype=cshogi.HuffmanCodedPosAndEval)
			board = cshogi.Board()
			logging.info(len(hcpes))
			for item in hcpes:
				board.set_hcp(item['hcp'])
				data = {}
				data['zobrist_hash'] = board.zobrist_hash()
				data['turn'] = board.turn
				data['sfen'] = board.sfen()
				data['CSAPos'] = board.csa_pos()
				data['eval'] = item['eval']
				koma = np.array(board.pieces, dtype=int)
				for sq in range(81):
					name = cshogi.SQUARE_NAMES[sq]
					if koma[sq] < 15:
						val = cshogi.PIECE_JAPANESE_SYMBOLS[koma[sq]]
					else:
						val = 'v' + cshogi.PIECE_JAPANESE_SYMBOLS[koma[sq] - 16]
					data[name] = val

				hand_koma_b = board.pieces_in_hand[cshogi.BLACK]
				hand_koma_w = board.pieces_in_hand[cshogi.WHITE]
				for i in range(7):
					data[cshogi.HAND_PIECE_JAPANESE_SYMBOLS[i]] = hand_koma_b[i]
					data['v' + cshogi.HAND_PIECE_JAPANESE_SYMBOLS[i]] = hand_koma_w[i]

				writer.writerow(data)

hcpe_to_csv(1)