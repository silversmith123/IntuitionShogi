import shogi
from shogi import CSA
import model
import search
import numpy as np
import glob
import itertools
import logging

### 初期局面
def test_case1():
	m_model = model.IntuitionModel()
	m_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	csa_name = "./kif/2017/wdoor+floodgate-300-10F+3Aeval-_Ryzen5-1600+gpsfish_normal_1c+20171204213002.csa"
	kif = CSA.Parser().parse_str(open(csa_name).read())
	board = shogi.Board(kif[0]['sfen'])
	best_move = kif[0]['moves'][0]

	target_features = []
	for move in shogi.LegalMoveGenerator(board):
		if move.usi() == best_move:
			continue
		board.push(move)
		#ponder_moves = search.ponder(board, m_model)
		#board.push(ponder_moves)
		target_features.append(model.board_to_features(board))
		board.pop()
	board.push(shogi.Move.from_usi(best_move))
	best_feature = model.board_to_features(board)

	target_features = model.ds_from_features(target_features, best_feature)
	#best_feature = model.ds_from_features(best_feature)

	m_model.fit(target_features)

### 1局
def test_case2():
	m_model = model.IntuitionModel()
	m_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	csa_name = "./kif/2017/wdoor+floodgate-300-10F+3Aeval-_Ryzen5-1600+gpsfish_normal_1c+20171204213002.csa"
	kif = CSA.Parser().parse_str(open(csa_name).read())
	board = shogi.Board(kif[0]['sfen'])
	target_features = []
	best_features = []
	for best_move in kif[0]['moves']:
		for move in shogi.LegalMoveGenerator(board):
			move_target = []
			if move.usi() == best_move:
				continue
			board.push(move)
			#ponder_moves = search.ponder(board, m_model)
			#board.push(ponder_moves)
			move_target.append(model.board_to_features(board))
			board.pop()
		board.push(shogi.Move.from_usi(best_move))
		target_features.append(move_target)
		best_features.append(model.board_to_features(board))

	target_features = model.ds_from_features2(target_features, best_features)
		#best_feature = model.ds_from_features(best_feature)

	m_model.fit(target_features, epochs=100)

### 複数局
def test_case3():
	m_model = model.IntuitionModel()
	m_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	target_features = []
	best_features = []
	for csa_name in itertools.islice(glob.glob("./kif/*/*.csa"), 20):
		logging.warning(csa_name)
		kif = CSA.Parser().parse_str(open(csa_name).read())
		board = shogi.Board(kif[0]['sfen'])
		for best_move in kif[0]['moves']:
			for move in shogi.LegalMoveGenerator(board):
				move_target = []
				if move.usi() == best_move:
					continue
				board.push(move)
				move_target.append(model.board_to_features(board))
				board.pop()
			board.push(shogi.Move.from_usi(best_move))
			if len(move_target) == 0:
				continue
			target_features.append(move_target)
			best_features.append(model.board_to_features(board))

	target_features = model.ds_from_features2(target_features, best_features)

	m_model.fit(target_features, epochs=100)

test_case3()