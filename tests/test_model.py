import CUI
import shogi

### 初期局面
def test_case1():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の歩が1枚足りない
def test_case2():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b p 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の香が1枚足りない
def test_case3():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/1NSGKGSNL b l 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の桂が1枚足りない
def test_case4():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/L1SGKGSNL b n 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の銀が1枚足りない
def test_case5():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LN1GKGSNL b s 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の金が1枚足りない
def test_case6():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNS1KGSNL b g 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の飛が1枚足りない
def test_case7():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B7/LNSGKGSNL b r 1"
	cui.board = shogi.Board(sfen)
	cui.show()
### 先手の角が1枚足りない
def test_case8():
	cui = CUI.CUI()
	sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL b b 1"
	cui.board = shogi.Board(sfen)
	cui.show()

test_case1()
test_case2()
test_case3()
test_case4()
test_case5()
test_case6()
test_case7()
test_case8()
