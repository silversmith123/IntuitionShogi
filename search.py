import model as m
import shogi
import sys
import queue

max_depth = 3

#@profile
def negamax(board, model, depth):
	score = -sys.float_info.max
	if depth >= max_depth:
		score = m.evaluate(board, model)
		return None, score
	for move in shogi.LegalMoveGenerator(board):
		board.push(move)
		_, update_score = negamax(board, model, depth + 1)
		update_score = -update_score
		if score < update_score:
			bestmove = move
			score = update_score
		board.pop()
	return bestmove, score

def score(board, model):
	#input_q = queue.queue()
	return negamax(board, model, 1)

def ponder(board, model):
	ret, _ = negamax(board, model, 1)
	return ret