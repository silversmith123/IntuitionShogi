
import preprocessing
from shogi import CSA
import shogi
import numpy as np
import tensorflow as tf

class CUI:

	def __init__(self):
		self.model = tf.keras.models.load_model('./model/intuitionshogi.pb')

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
		eval = float(self.model({'inputs_koma_b': inputs_koma_b, 'inputs_koma_w': inputs_koma_w, 'inputs_hand_koma_b':inputs_hand_koma_b, 'inputs_hand_koma_w':inputs_hand_koma_w, 'inputs_atk_koma_b': inputs_atk_koma_b, 'inputs_atk_koma_w': inputs_atk_koma_w, 'inputs_difence_koma_b':inputs_difence_koma_b, 'inputs_difence_koma_w':inputs_difence_koma_w}, training=False))
		if self.board.turn == shogi.WHITE:
			eval = -eval
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

	def show(self):
		print(self.board.kif_str())
		print(self.eval())
		print('')

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

cui = CUI()
cui.main()