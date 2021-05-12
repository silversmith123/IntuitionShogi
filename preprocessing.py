import numpy as np
import tensorflow as tf
import shogi

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
    atk_koma_b = np.zeros((81, 15), dtype=float)
    atk_koma_w = np.zeros((81, 15), dtype=float)
    difence_koma_b = np.zeros((81, 15), dtype=float)
    difence_koma_w = np.zeros((81, 15), dtype=float)

    for sq in range(81):
        for atk_sq in board.attackers(shogi.BLACK, sq):
            if board.piece_at(atk_sq):
                atk_koma_b[sq][board.piece_at(atk_sq).piece_type] = 1
            if board.piece_at(atk_sq):
                difence_koma_b[sq][board.piece_at(atk_sq).piece_type] = 1
        for atk_sq in board.attackers(shogi.WHITE, sq):
            if board.piece_at(atk_sq):
                atk_koma_w[sq][board.piece_at(atk_sq).piece_type] = 1
            if board.piece_at(atk_sq):
                difence_koma_w[sq][board.piece_at(atk_sq).piece_type] = 1

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
    features['difence_koma_b'] = difence_koma_b
    features['difence_koma_w'] = difence_koma_w
    # features['eval'].append(eval)
    # features['gameResult'].append(gameResult)
    # features['turn']=np.array(features['turn'])
    features['koma_b'] = np.array(features['koma_b'], dtype=float)
    features['koma_w'] = np.array(features['koma_w'], dtype=float)
    features['atk_koma_b'] = np.array(features['atk_koma_b'], dtype=float)
    features['atk_koma_w'] = np.array(features['atk_koma_w'], dtype=float)
    features['difence_koma_b'] = np.array(features['difence_koma_b'], dtype=float)
    features['difence_koma_w'] = np.array(features['difence_koma_w'], dtype=float)
    features['hand_koma_b'] = np.array(features['hand_koma_b'], dtype=float)
    features['hand_koma_w'] = np.array(features['hand_koma_w'], dtype=float)
    # features['eval']=np.array(features['eval'])
    # features['gameResult']=np.array(features['gameResult'])

    return features

def ds_from_features(features, y_features):
	koma_b = []
	koma_w = []
	atk_koma_b = []
	atk_koma_w = []
	hand_koma_b = []
	hand_koma_w = []
	difence_koma_b = []
	difence_koma_w = []

	y_koma_b = []
	y_koma_w = []
	y_atk_koma_b = []
	y_atk_koma_w = []
	y_hand_koma_b = []
	y_hand_koma_w = []
	y_difence_koma_b = []
	y_difence_koma_w = []

	for move_feature, best_feature in zip(features, y_features):
		tgt = {}
		tgt['koma_b'] = []
		tgt['koma_w'] = []
		tgt['atk_koma_b'] = []
		tgt['atk_koma_w'] = []
		tgt['hand_koma_b'] = []
		tgt['hand_koma_w'] = []
		tgt['difence_koma_b'] = []
		tgt['difence_koma_w'] = []
		tgt['y_koma_b'] = []
		tgt['y_koma_w'] = []
		tgt['y_atk_koma_b'] = []
		tgt['y_atk_koma_w'] = []
		tgt['y_hand_koma_b'] = []
		tgt['y_hand_koma_w'] = []
		tgt['y_difence_koma_b'] = []
		tgt['y_difence_koma_w'] = []

		for f in move_feature:
			tgt['koma_b'].append(f['koma_b'])
			tgt['koma_w'].append(f['koma_w'])
			tgt['atk_koma_b'].append(f['atk_koma_b'])
			tgt['atk_koma_w'].append(f['atk_koma_w'])
			tgt['hand_koma_b'].append(f['hand_koma_b'])
			tgt['hand_koma_w'].append(f['hand_koma_w'])
			tgt['difence_koma_b'].append(f['difence_koma_b'])
			tgt['difence_koma_w'].append(f['difence_koma_w'])

		tgt['y_koma_b'] = best_feature['koma_b']
		tgt['y_koma_w'] = best_feature['koma_w']
		tgt['y_atk_koma_b'] = best_feature['atk_koma_b']
		tgt['y_atk_koma_w'] = best_feature['atk_koma_w']
		tgt['y_hand_koma_b'] = best_feature['hand_koma_b']
		tgt['y_hand_koma_w'] = best_feature['hand_koma_w']
		tgt['y_difence_koma_b'] = best_feature['difence_koma_b']
		tgt['y_difence_koma_w'] = best_feature['difence_koma_w']
	
		koma_b.append(tgt['koma_b'])
		koma_w.append(tgt['koma_w'])
		atk_koma_b.append(tgt['atk_koma_b'])
		atk_koma_w.append(tgt['atk_koma_w'])
		hand_koma_b.append(tgt['hand_koma_b'])
		hand_koma_w.append(tgt['hand_koma_w'])
		difence_koma_b.append(tgt['difence_koma_b'])
		difence_koma_w.append(tgt['difence_koma_w'])
	
		y_koma_b.append(tgt['y_koma_b'])
		y_koma_w.append(tgt['y_koma_w'])
		y_atk_koma_b.append(tgt['y_atk_koma_b'])
		y_atk_koma_w.append(tgt['y_atk_koma_w'])
		y_hand_koma_b.append(tgt['y_hand_koma_b'])
		y_hand_koma_w.append(tgt['y_hand_koma_w'])
		y_difence_koma_b.append(tgt['y_difence_koma_b'])
		y_difence_koma_w.append(tgt['y_difence_koma_w'])

	koma_b = np.array(koma_b, dtype=float)
	koma_w = np.array(koma_w, dtype=float)
	atk_koma_b = np.array(atk_koma_b, dtype=float)
	atk_koma_w = np.array(atk_koma_w, dtype=float)
	hand_koma_b = np.array(hand_koma_b, dtype=float)
	hand_koma_w = np.array(hand_koma_w, dtype=float)
	difence_koma_b = np.array(difence_koma_b, dtype=float)
	difence_koma_w = np.array(difence_koma_w, dtype=float)

	y_koma_b = np.array(y_koma_b, dtype=float)
	y_koma_w = np.array(y_koma_w, dtype=float)
	y_atk_koma_b = np.array(y_atk_koma_b, dtype=float)
	y_atk_koma_w = np.array(y_atk_koma_w, dtype=float)
	y_hand_koma_b = np.array(y_hand_koma_b, dtype=float)
	y_hand_koma_w = np.array(y_hand_koma_w, dtype=float)
	y_difence_koma_b = np.array(y_difence_koma_b, dtype=float)
	y_difence_koma_w = np.array(y_difence_koma_w, dtype=float)


	dataset = tf.data.Dataset.from_tensor_slices(({'koma_b': koma_b, 'koma_w': koma_w, 'atk_koma_b': atk_koma_b, 'atk_koma_w': atk_koma_w, 'hand_koma_b': hand_koma_b, 'hand_koma_w': hand_koma_w, 'difence_koma_b': difence_koma_b, 'difence_koma_w': difence_koma_w, 'koma_b': koma_b, 'y_koma_b': y_koma_b, 'y_koma_w': y_koma_w, 'y_atk_koma_b': y_atk_koma_b, 'y_atk_koma_w': y_atk_koma_w, 'y_hand_koma_b': y_hand_koma_b, 'y_hand_koma_w': y_hand_koma_w, 'y_difence_koma_b': y_difence_koma_b, 'y_difence_koma_w': y_difence_koma_w}))
	return dataset