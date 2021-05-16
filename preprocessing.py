import numpy as np
import tensorflow as tf
import shogi
from tensorflow.python import profiler

def swap(features):
	# 持ち駒は交換するだけ
	koma_b = features['koma_b']
	koma_w = features['koma_w']
	hand_koma_b = features['hand_koma_b']
	hand_koma_w = features['hand_koma_w']
	atk_koma_b = features['atk_koma_b']
	atk_koma_w = features['atk_koma_w']

	hand_koma_b, hand_koma_w = hand_koma_w, hand_koma_b
	# 盤上の駒は交換後、位置をひっくり返す
	koma_b, koma_w = koma_w, koma_b
	koma_b = np.array([koma_b[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]] for sq in range(81)])
	koma_w = np.array([koma_w[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]] for sq in range(81)])
	atk_koma_b, atk_koma_w = atk_koma_w, atk_koma_b
	atk_koma_b = np.array([atk_koma_b[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]]
							for sq in range(81)])
	atk_koma_w = np.array([atk_koma_w[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]]
							for sq in range(81)])

	features['koma_b'] = koma_b
	features['koma_w'] = koma_w
	features['hand_koma_b'] = hand_koma_b
	features['hand_koma_w'] = hand_koma_w
	features['atk_koma_b'] = atk_koma_b
	features['atk_koma_w'] = atk_koma_w

	return features

def board_to_features(board):
    # 駒の配列
    koma_b = [board.piece_at(sq).piece_type if board.piece_at(
        sq) != None and board.piece_at(sq).color == shogi.BLACK else 0 for sq in range(81)]
    koma_w = [board.piece_at(sq).piece_type if board.piece_at(
        sq) != None and board.piece_at(sq).color == shogi.WHITE else 0 for sq in range(81)]
    koma_b = np.array(koma_b)
    koma_b = np.eye(15)[koma_b.reshape(-1)]
    koma_w = np.array(koma_w)
    koma_w = np.eye(15)[koma_w.reshape(-1)]
    # 攻撃している駒の配列
    atk_koma_b = np.zeros((81, 15), dtype=np.float32)
    atk_koma_w = np.zeros((81, 15), dtype=np.float32)

    for sq in range(81):
        for atk_sq in board.attackers(shogi.BLACK, sq):
            atk_koma_b[sq][board.piece_type_at(atk_sq)] += 1
        for atk_sq in board.attackers(shogi.WHITE, sq):
            atk_koma_w[sq][board.piece_type_at(atk_sq)] += 1

        # 持ち駒
    hand_koma_b = [0] * 7
    p = board.pieces_in_hand[shogi.BLACK]
    for piece_type in shogi.PIECE_TYPES[0:7]:
        hand_koma_b[shogi.PIECE_TYPES.index(piece_type)] = p[piece_type]
    hand_koma_w = [0] * 7
    p = board.pieces_in_hand[shogi.WHITE]
    for piece_type in shogi.PIECE_TYPES[0:7]:
        hand_koma_w[shogi.PIECE_TYPES.index(piece_type)] = p[piece_type]

        # 全て先手番から見た位置に変換
    if board.turn == shogi.WHITE:
        # 持ち駒は交換するだけ
        hand_koma_b, hand_koma_w = hand_koma_w, hand_koma_b
        # 盤上の駒は交換後、位置をひっくり返す
        koma_b, koma_w = koma_w, koma_b
        koma_b = np.array([koma_b[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]] for sq in range(81)])
        koma_w = np.array([koma_w[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]] for sq in range(81)])
        atk_koma_b, atk_koma_w = atk_koma_w, atk_koma_b
        atk_koma_b = np.array([atk_koma_b[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]]
                               for sq in range(81)])
        atk_koma_w = np.array([atk_koma_w[shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]]
                               for sq in range(81)])
    # 9*9にrehshape
    #koma_b = koma_b.reshape(9, 9, 15)
    #koma_w = koma_w.reshape(9, 9, 15)
    #atk_koma_b = atk_koma_b.reshape(9, 9, 15)
    #atk_koma_w = atk_koma_w.reshape(9, 9, 15)
    # features['turn'].append(board.turn)
    features = {}
    features['koma_b'] = koma_b
    features['koma_w'] = koma_w
    features['atk_koma_b'] = atk_koma_b
    features['atk_koma_w'] = atk_koma_w
    features['hand_koma_b'] = hand_koma_b
    features['hand_koma_w'] = hand_koma_w
    # features['eval'].append(eval)
    # features['gameResult'].append(gameResult)
    # features['turn']=np.array(features['turn'])
    features['koma_b'] = np.array(features['koma_b'], dtype=np.float32)
    features['koma_w'] = np.array(features['koma_w'], dtype=np.float32)
    features['atk_koma_b'] = np.array(features['atk_koma_b'], dtype=np.float32)
    features['atk_koma_w'] = np.array(features['atk_koma_w'], dtype=np.float32)
    features['hand_koma_b'] = np.array(features['hand_koma_b'], dtype=np.float32)
    features['hand_koma_w'] = np.array(features['hand_koma_w'], dtype=np.float32)
    # features['eval']=np.array(features['eval'])
    # features['gameResult']=np.array(features['gameResult'])

    return features

def update_features(board, move, features):
	def reverse(sq):
		return shogi.SQUARES_L90[shogi.SQUARES_L90[sq]]

	def reset_atk_feature(piece_type, square, occupied, color, feature):
		moves = board.attacks_from(piece_type, square, occupied, color)
		sq = shogi.bit_scan(moves)
		while sq != -1 and sq is not None:
			if color == shogi.BLACK and board.turn == shogi.BLACK:
				feature[sq][piece_type] -= 1
			elif color == shogi.WHITE and board.turn == shogi.BLACK:
				feature[sq][piece_type] -= 1
			elif color == shogi.BLACK and board.turn == shogi.WHITE:
				feature[reverse(sq)][piece_type] -= 1
			else:
				feature[reverse(sq)][piece_type] -= 1
			sq = shogi.bit_scan(moves, sq + 1)
		return feature

	def add_atk_feature(piece_type, square, occupied, color, feature):
		moves = board.attacks_from(piece_type, square, occupied, color)
		sq = shogi.bit_scan(moves)
		while sq != -1 and sq is not None:
			if color == shogi.BLACK and board.turn == shogi.BLACK:
				feature[sq][piece_type] += 1
			elif color == shogi.WHITE and board.turn == shogi.BLACK:
				feature[sq][piece_type] += 1
			elif color == shogi.BLACK and board.turn == shogi.WHITE:
				feature[reverse(sq)][piece_type] += 1
			else:
				feature[reverse(sq)][piece_type] += 1
			sq = shogi.bit_scan(moves, sq + 1)
		return feature

	if move.from_square:
		move_koma = board.piece_at(move.from_square)
	else:
		move_koma = None
	# 駒の位置 移動元の駒は無くなる。移動先の駒は置き換えられる。
	if move_koma:
		if board.turn == shogi.BLACK:
			features['koma_b'][move.from_square][board.piece_type_at(move.from_square)] = 0
		else:
			features['koma_b'][reverse(move.from_square)][board.piece_type_at(move.from_square)] = 0

	# コマの攻撃
	# コマを動かす手
	if move_koma:
		# 元々攻撃していたマスを削除
		features['atk_koma_b'] = reset_atk_feature(move_koma.piece_type, move.from_square, board.occupied, board.turn, features['atk_koma_b'])
		# 元のマス目を攻撃していた敵のコマを削除
		move_koma_attackers = board.attackers(board.turn ^ 1, move.from_square)
		for sq in move_koma_attackers:
			features['atk_koma_w'] = reset_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn ^ 1, features['atk_koma_w'])
		# 元のマス目を攻撃していた味方のコマを削除
		move_koma_ally_attackers = board.attackers(board.turn, move.from_square)
		for sq in move_koma_ally_attackers:
			features['atk_koma_b'] = reset_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn, features['atk_koma_b'])

	# 取られるコマが攻撃していたマスを削除
	atked_koma = board.piece_at(move.to_square)
	if atked_koma:
		features['atk_koma_b'] = reset_atk_feature(atked_koma.piece_type, move.from_square, board.occupied, board.turn, features['atk_koma_b'])
	# 取られるマスを攻撃していた敵の駒を削除
	atked_koma_attackers = board.attackers(board.turn ^ 1, move.to_square)
	for sq in atked_koma_attackers:
		features['atk_koma_w'] = reset_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn ^ 1, features['atk_koma_w'])
	# 取られるマスを攻撃していた味方の駒を削除
	atked_koma_ally_attackers = board.attackers(board.turn, move.to_square)
	for sq in atked_koma_ally_attackers:
		if sq == move.from_square:
			continue
		features['atk_koma_b'] = reset_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn, features['atk_koma_b'])

	# 盤面を動かす
	board.push(move)
	features = swap(features)
	# 駒の位置 移動元の駒は無くなる。移動先の駒は置き換えられる。
	moved_koma = board.piece_at(move.to_square)
	if board.turn == shogi.WHITE:
		features['koma_w'][reverse(move.to_square)][moved_koma.piece_type] = 1
	else:
		features['koma_w'][move.to_square][moved_koma.piece_type] = 1

	# 持ち駒 先後関係なく更新
	p_b= board.pieces_in_hand[shogi.BLACK]
	p_w= board.pieces_in_hand[shogi.WHITE]

	from_hand_type = moved_koma.piece_type
	if from_hand_type >= shogi.PROM_PAWN:
		from_hand_type = shogi.PIECE_PROMOTED.index(from_hand_type)
	if from_hand_type < shogi.KING:
		features['hand_koma_b'][shogi.PIECE_TYPES.index(from_hand_type)] = p_b[from_hand_type]
		features['hand_koma_w'][shogi.PIECE_TYPES.index(from_hand_type)] = p_w[from_hand_type]

	if atked_koma:
		to_hand_type = atked_koma.piece_type
		if to_hand_type >= shogi.PROM_PAWN:
			to_hand_type = shogi.PIECE_PROMOTED.index(to_hand_type)
		if to_hand_type < shogi.KING:
			features['hand_koma_b'][shogi.PIECE_TYPES.index(to_hand_type)] = p_b[to_hand_type]
			features['hand_koma_w'][shogi.PIECE_TYPES.index(to_hand_type)] = p_w[to_hand_type]

	# コマの攻撃 注 取られた駒の攻撃はない
	# 移動後のコマの攻撃
	features['atk_koma_w'] = add_atk_feature(moved_koma.piece_type, move.to_square, board.occupied, board.turn ^ 1, features['atk_koma_w'])
	# 元のマス目を攻撃していた敵のコマを追加
	if move_koma:
		for sq in move_koma_attackers:
			features['atk_koma_b'] = add_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn, features['atk_koma_b'])
		# 元のマス目を攻撃していた味方のコマを追加
		for sq in move_koma_ally_attackers:
			if sq == move.to_square:
				continue
			features['atk_koma_w'] = add_atk_feature(board.piece_type_at(sq), sq, board.occupied, board.turn ^ 1, features['atk_koma_w'])

	return features


def ds_from_features(features, y_features):
	koma_b = []
	koma_w = []
	atk_koma_b = []
	atk_koma_w = []
	hand_koma_b = []
	hand_koma_w = []

	y_koma_b = []
	y_koma_w = []
	y_atk_koma_b = []
	y_atk_koma_w = []
	y_hand_koma_b = []
	y_hand_koma_w = []

	for move_feature, best_feature in zip(features, y_features):
		tgt = {}
		tgt['koma_b'] = []
		tgt['koma_w'] = []
		tgt['atk_koma_b'] = []
		tgt['atk_koma_w'] = []
		tgt['hand_koma_b'] = []
		tgt['hand_koma_w'] = []
		tgt['y_koma_b'] = []
		tgt['y_koma_w'] = []
		tgt['y_atk_koma_b'] = []
		tgt['y_atk_koma_w'] = []
		tgt['y_hand_koma_b'] = []
		tgt['y_hand_koma_w'] = []

		for f in move_feature:
			tgt['koma_b'].append(f['koma_b'])
			tgt['koma_w'].append(f['koma_w'])
			tgt['atk_koma_b'].append(f['atk_koma_b'])
			tgt['atk_koma_w'].append(f['atk_koma_w'])
			tgt['hand_koma_b'].append(f['hand_koma_b'])
			tgt['hand_koma_w'].append(f['hand_koma_w'])

		tgt['y_koma_b'] = best_feature['koma_b']
		tgt['y_koma_w'] = best_feature['koma_w']
		tgt['y_atk_koma_b'] = best_feature['atk_koma_b']
		tgt['y_atk_koma_w'] = best_feature['atk_koma_w']
		tgt['y_hand_koma_b'] = best_feature['hand_koma_b']
		tgt['y_hand_koma_w'] = best_feature['hand_koma_w']
	
		koma_b.append(tgt['koma_b'])
		koma_w.append(tgt['koma_w'])
		atk_koma_b.append(tgt['atk_koma_b'])
		atk_koma_w.append(tgt['atk_koma_w'])
		hand_koma_b.append(tgt['hand_koma_b'])
		hand_koma_w.append(tgt['hand_koma_w'])
	
		y_koma_b.append(tgt['y_koma_b'])
		y_koma_w.append(tgt['y_koma_w'])
		y_atk_koma_b.append(tgt['y_atk_koma_b'])
		y_atk_koma_w.append(tgt['y_atk_koma_w'])
		y_hand_koma_b.append(tgt['y_hand_koma_b'])
		y_hand_koma_w.append(tgt['y_hand_koma_w'])

	koma_b = np.array(koma_b, dtype=float)
	koma_w = np.array(koma_w, dtype=float)
	atk_koma_b = np.array(atk_koma_b, dtype=float)
	atk_koma_w = np.array(atk_koma_w, dtype=float)
	hand_koma_b = np.array(hand_koma_b, dtype=float)
	hand_koma_w = np.array(hand_koma_w, dtype=float)

	y_koma_b = np.array(y_koma_b, dtype=float)
	y_koma_w = np.array(y_koma_w, dtype=float)
	y_atk_koma_b = np.array(y_atk_koma_b, dtype=float)
	y_atk_koma_w = np.array(y_atk_koma_w, dtype=float)
	y_hand_koma_b = np.array(y_hand_koma_b, dtype=float)
	y_hand_koma_w = np.array(y_hand_koma_w, dtype=float)


	dataset = tf.data.Dataset.from_tensor_slices(({'koma_b': koma_b, 'koma_w': koma_w, 'atk_koma_b': atk_koma_b, 'atk_koma_w': atk_koma_w, 'hand_koma_b': hand_koma_b, 'hand_koma_w': hand_koma_w, 'y_koma_b': y_koma_b, 'y_koma_w': y_koma_w, 'y_atk_koma_b': y_atk_koma_b, 'y_atk_koma_w': y_atk_koma_w, 'y_hand_koma_b': y_hand_koma_b, 'y_hand_koma_w': y_hand_koma_w}))
	return dataset