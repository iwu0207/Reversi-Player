# I pledge on my honor that I have not given or received any unauthorized assistance on this project.
# Isaac Wu
import math

from reversi import *
import supervisor
import random
import time

Seen = []
reward = [[0] * 8] * 8
times = [[1] * 8] * 8
other_tile = 'X'
depth = 0
max_depth = 3


def get_move(board, tile):
    possible_moves = getValidMoves(board, tile)
    if tile == 'X':
        global other_tile
        other_tile = 'O'

    # print(tile)
    #best_move = possible_moves[0]
    #best = uct(possible_moves[0], board, tile, True)

    for x, y in possible_moves:  # from computer's strategy
        if isOnCorner(x, y):
            return [x, y]

    for play in possible_moves:
        for i in range(45):
            new_board = getBoardCopy(board)
            # print(play)
            #global Depth
            #Depth = 0
            uct(play, new_board, tile, True)
            #    best = uct(play, newboard, tile, True)
            #    best_move = play  # updates rewards and times global variables to calc Q later

    best_move = possible_moves[0]

    for play in possible_moves:  # calc Q value for each possible action and compare best ones
        #q1 = reward[best_move[0] - 1][best_move[1] - 1] + math.sqrt(
        #    2 * math.log(sum(sum(times, [])), math.e) / times[best_move[0] - 1][best_move[1] - 1])
        #q2 = reward[play[0] - 1][play[1] - 1] + math.sqrt(
        #    2 * math.log(sum(sum(times, [])), math.e) / times[play[0] - 1][play[1] - 1])
        q2 = reward[play[0] - 1][play[1] - 1]
        q1 = reward[best_move[0] - 1][best_move[1] - 1]

        if q2 > q1:
            best_move = play

    return best_move  # returns best move in [x,y] according to the highest rewards


def moveinlist(ls, mo):  # is the move mo in list ls
    if ls is not [] and mo is not None:
        for i in ls:
            if i is not None and i[0] == mo[0] and i[1] == mo[1]:
                return True
    return False


# global Seen <- null
def uct(move_uct, board, tile, plyr1_turn):  # function UCT(x)
    v = 0

    global depth
    depth += 1

    newboard = getBoardCopy(board)
    if move_uct is None:
        v = getScoreOfBoard(newboard)[tile]
        return v
    if not moveinlist(Seen, move_uct):  # x not an elem of Seen
        Seen.append(move_uct)  # add x to Seen
        reward[move_uct[0] - 1][move_uct[1] - 1] = 0  # rx <- 0
        times[move_uct[0] - 1][move_uct[1] - 1] = 1  # tx <- 0
    if plyr1_turn:
        makeMove(newboard, tile, move_uct[0], move_uct[1])  # make the move, moveuct on a temp board
    else:
        makeMove(newboard, other_tile, move_uct[0], move_uct[1])
    # print(newboard)
    # print(getScoreOfBoard(newboard))
    if depth > max_depth:
        return getScoreOfBoard(newboard)[tile]
    if plyr1_turn is True:
        turn_tile = other_tile
    else:
        turn_tile = tile
    if getValidMoves(newboard, turn_tile) == []: # if x is a terminal node(no valid moves)
        v = getScoreOfBoard(newboard)[tile]

    else:
        if plyr1_turn:
            child = ucb_choose(move_uct, newboard, other_tile, not plyr1_turn)  # max child or min child
            v = uct(child, newboard, other_tile, not plyr1_turn)
        else:
            child = ucb_choose(move_uct, newboard, tile, plyr1_turn)  # max child or min child
            v = uct(child, newboard, tile, plyr1_turn)
    if plyr1_turn:
        reward[move_uct[0] - 1][move_uct[1] - 1] = ((reward[move_uct[0] - 1][move_uct[1] - 1] * times[move_uct[0] - 1][
            move_uct[1] - 1]) + v) / (times[move_uct[0] - 1][move_uct[1] - 1] + 1)
        times[move_uct[0] - 1][move_uct[1] - 1] = times[move_uct[0] - 1][move_uct[1] - 1] + 1
    else:
        reward[move_uct[0] - 1][move_uct[1] - 1] = ((reward[move_uct[0] - 1][move_uct[1] - 1] * times[move_uct[0] - 1][
            move_uct[1] - 1]) + v) / (times[move_uct[0] - 1][move_uct[1] - 1] + 1)
        times[move_uct[0] - 1][move_uct[1] - 1] = times[move_uct[0] - 1][move_uct[1] - 1] + 1
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
            t=t+1

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
