
import preprocessing
import logging
from shogi import CSA
import shogi
import numpy as np
import tensorflow as tf
import model

class CUI:

	def __init__(self):
		self.debug = True
		self.model = model.IntuitionModel()
		self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"], run_eagerly=True)

	def eval(self):
		features = preprocessing.board_to_features(self.board)
		inputs_koma_b = np.array([features['koma_b']])
		inputs_koma_w = np.array([features['koma_w']])
		inputs_hand_koma_b = np.array([features['hand_koma_b']])
		inputs_hand_koma_w = np.array([features['hand_koma_w']])
		inputs_atk_koma_b = np.array([features['atk_koma_b']])
		inputs_atk_koma_w = np.array([features['atk_koma_w']])
		inputs_difence_koma_b = np.array([features['difence_koma_b']])
		inputs_difence_koma_w = np.array([features['difence_koma_w']])
		eval = float(self.model.call({'inputs_koma_b': inputs_koma_b, 'inputs_koma_w': inputs_koma_w, 'inputs_hand_koma_b':inputs_hand_koma_b, 'inputs_hand_koma_w':inputs_hand_koma_w, 'inputs_atk_koma_b': inputs_atk_koma_b, 'inputs_atk_koma_w': inputs_atk_koma_w, 'inputs_difence_koma_b':inputs_difence_koma_b, 'inputs_difence_koma_w':inputs_difence_koma_w}))
		return eval

	def forward(self, move):
		self.board.push(move)

	def back(self):
		self.board.pop()

	def input(self, action):
		if action == "back":
			self.back()
		else:
			move = shogi.Move.from_usi(action)
			self.forward(move)

	def show_feature(self):
		features = preprocessing.board_to_features(self.board)
		def _print(feature):
			for rank in range(9):
				line = ''
				for col in range(9):
					cell = 9 * rank + col
					val = 0
					for type in shogi.PIECE_TYPES:
						val += int(feature[cell][type]) * type
					line += ' ' +str(val)
				print(line)

		for type in ['koma_b', 'koma_w', 'atk_koma_b', 'atk_koma_w', 'difence_koma_b', 'difence_koma_w']:
			print(type)
			_print(features[type])

		for type in ['hand_koma_b', 'hand_koma_w']:
			print(type)
			line = ''
			for koma in features[type]:
				line += str(int(koma))
			print(line)

	def show(self):
		if self.debug:
			self.show_feature()
		print(self.board.kif_str())
		if self.board.turn == shogi.BLACK:
			print('先手番')
		else:
			print('後手番')
		print('評価値:' + str(self.eval()))

	def main(self):
		print('指し手の表記')
		print('移動 ７七から７六 7g7f')
		print('移動 + 成る ７八から２二 8h2b+')
		print('駒を打つ 5二銀打つ G*5b')
		print('段のアルフベット表現 一:a 二:b 三:c 四:d 五:e 六:f 七:g 八:h 九:i')
		print('駒の表現 玉:K 飛車:R 角:B 金:G 銀:S 桂馬:N 香車:L 歩:P')
		sfen = input('初期局面をsfenで入力してください。\n')
		self.board = shogi.Board(sfen)
		self.show()
		while True:
			action = input('1手進める場合は指し手、1手戻す場合はback、終了する場合はexitを入力してください。\n')
			if action == 'exit':
				break
			else:
				self.input(action)
			self.show()