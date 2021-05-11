import model
import shogi

### 初期局面
def test_case1():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の歩が1枚足りない
def test_case2():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b p 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の香が1枚足りない
def test_case3():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/1NSGKGSNL b l 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の桂が1枚足りない
def test_case4():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/L1SGKGSNL b n 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の銀が1枚足りない
def test_case5():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LN1GKGSNL b s 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の金が1枚足りない
def test_case6():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNS1KGSNL b g 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の飛が1枚足りない
def test_case7():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B7/LNSGKGSNL b r 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))
### 先手の角が1枚足りない
def test_case8():
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL b b 1"
    board = shogi.Board(sfen=sfen)
    print(board.kif_str()) 
    print(model.eval_sfen(sfen))

test_case1()
test_case2()
test_case3()
test_case4()
test_case5()
test_case6()
test_case7()
test_case8()
