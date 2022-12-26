# I pledge on my honor that I have not given or received any unauthorized assistance on this project.
# Isaac Wu
import math

from reversi import *
import supervisor
import random
import time

Seen = []
reward = [[0] * 8] * 8
times = [[0] * 8] * 8
other_tile = 'X'


def get_move(board, tile):
    possibleMoves = getValidMoves(board, tile)
    if tile == 'X':
        global other_tile
        other_tile = 'O'

    # print(tile)
    for play in possibleMoves:
        newboard = getBoardCopy(board)
        # print(play)
        uct(play, newboard, tile, True)  # updates rewards and times global variables to calc Q later

    bestmove = possibleMoves[0]

    for play in possibleMoves:  # calc Q value for each possible action and compare best ones
        q1 = reward[bestmove[0] - 1][bestmove[1] - 1] + math.sqrt(
            2 * math.log(sum(sum(times, [])), math.e) / times[bestmove[0] - 1][bestmove[1] - 1])
        q2 = reward[play[0] - 1][play[1] - 1] + math.sqrt(
            2 * math.log(sum(sum(times, [])), math.e) / times[play[0] - 1][play[1] - 1])

        if q2 > q1:
            bestmove = play

    return bestmove  # returns best move in [x,y] according to the highest rewards


def moveinlist(ls, mo):  # is the move mo in list ls
    if ls is not [] and mo is not None:
        for i in ls:
            if i is not None and i[0] == mo[0] and i[1] == mo[1]:
                return True
    return False


# global Seen <- null
def uct(moveuct, board, tile, plyr1turn):  # function UCT(x)
    v = 0
    newboard = getBoardCopy(board)
    if moveuct is None:
        v = getScoreOfBoard(newboard)[tile]
        return v
    if not moveinlist(Seen, moveuct):  # x not an elem of Seen
        Seen.append(moveuct)  # add x to Seen
        reward[moveuct[0] - 1][moveuct[1] - 1] = 0  # rx <- 0
        times[moveuct[0] - 1][moveuct[1] - 1] = 0  # tx <- 0
    if plyr1turn:
        makeMove(newboard, tile, moveuct[0], moveuct[1])  # make the move, moveuct on a temp board
    else:

        makeMove(newboard, tile, moveuct[0], moveuct[1])
    # print(newboard)
    # print(getScoreOfBoard(newboard))
    if getValidMoves(newboard, tile) == []:  # if x is a terminal node(no valid moves)
        v = getScoreOfBoard(newboard)[tile]# value based on diff of X and O

    else:
        if plyr1turn:
            child = ucb_choose(moveuct, newboard, other_tile, not plyr1turn)  # max child or min child
            v = uct(child, newboard, other_tile, not plyr1turn)
        else:
            child = ucb_choose(moveuct, newboard, tile, plyr1turn)  # max child or min child
            v = uct(child, newboard, tile, plyr1turn)

    reward[moveuct[0] - 1][moveuct[1] - 1] = ((reward[moveuct[0] - 1][moveuct[1] - 1] * times[moveuct[0] - 1][
        moveuct[1] - 1]) + v) / (times[moveuct[0] - 1][moveuct[1] - 1] + 1)
    times[moveuct[0] - 1][moveuct[1] - 1] = times[moveuct[0] - 1][moveuct[1] - 1] + 1
    return v


def ucb_choose(moveuct, board, tile, plyr1turn):
    if not plyr1turn:
        children = getValidMoves(board, other_tile)
    else:
        children = getValidMoves(board, tile)
    # print(children)
    yanSeen = []
    for i in children:
        for j in Seen:
            # if i[0] == j[0] and i[1] == j[1]:
            if i == j:
                yanSeen.append(i)

    if yanSeen == []:
        random.shuffle(children)
        if children == []:
            return moveuct
        return children[0]

    t = 0

    for i in children:
        t += times[i[0] - 1][i[1] - 1]

    if plyr1turn:  # if plyr1turn, maximize Max Payoff
        maxy = None
        maxall = 0
        for i in children:
            newboard = getBoardCopy(board)
            makeMove(newboard, tile, i[0], i[1])
            score = getScoreOfBoard(newboard)[tile]
            reward[i[0] - 1][i[1] - 1] = ((reward[i[0] - 1][i[1] - 1] * times[i[0] - 1][
                i[1] - 1]) + score) / (times[i[0] - 1][i[1] - 1] + 1)
            times[i[0] - 1][i[1] - 1] = times[i[0] - 1][i[1] - 1]+1
            t = t+1

        for i in children:
            if times[i[0] - 1][i[1] - 1] is not 0:
                argmax = reward[i[0] - 1][i[1] - 1] + math.sqrt(2 * math.log(t, math.e) / times[i[0] - 1][i[1] - 1])
                if argmax > maxall:
                    maxall = argmax
                    maxy = i
            # newboard = getBoardCopy(board)
            # makeMove(newboard, tile, i[0], i[1])
            # score = getScoreOfBoard(newboard)[tile]
            # if score > maxall:
            #    maxall = score
            #    maxy = i

        return maxy

    else:  # maximize Min payoff
        maxy = None
        maxall = 0
        for i in children:
            newboard = getBoardCopy(board)
            makeMove(newboard, tile, i[0], i[1])
            score = getScoreOfBoard(newboard)[tile]
            reward[i[0] - 1][i[1] - 1] = ((reward[i[0] - 1][i[1] - 1] * times[i[0] - 1][
                i[1] - 1]) + score) / (times[i[0] - 1][i[1] - 1] + 1)
            times[i[0] - 1][i[1] - 1] = times[i[0] - 1][i[1] - 1] + 1
            t = t+1

            if times[i[0] - 1][i[1] - 1] is not 0:
                argmax = 1 - reward[i[0] - 1][i[1] - 1] + math.sqrt(2 * math.log(t, math.e) / times[i[0] - 1][i[1] - 1])
                if argmax > maxall:
                    maxall = argmax
                    maxy = i


            # if score < maxall:
            #    maxall = score
            #    maxy = i

        return maxy
