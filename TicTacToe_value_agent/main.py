from enum import Enum
from turtle import circle
import random

class Players(Enum):
    empty = 0
    circle = 1
    cross = 2

CROSS = 0
CIRCLE = 1



class State:
    def __init__(self):
        self.board = [
            [Players.empty, Players.empty, Players.empty],
            [Players.empty, Players.empty, Players.empty],
            [Players.empty, Players.empty, Players.empty]
        ]
        self.turn = Players.cross
        self.moves_made = 0

    def switch_turns(self):
        if self.turn == Players.cross:
            self.turn = Players.circle
        else:
            self.turn = Players.cross


    def is_terminal(self):
        for i in range(3):
            if self.board[i][0] != Players.empty and self.board[i][0] == self.board[i][1] and self.board[i][1] == self.board[i][2]:
                return True, self.board[i][2]
            if self.board[0][i] != Players.empty and self.board[0][i] == self.board[1][i] and self.board[1][i] == self.board[2][i]:
                return True, self.board[2][i]

        if self.board[0][0] != Players.empty and self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]:
            return True, self.board[2][2]

        if self.board[0][2] != Players.empty and self.board[0][2] == self.board[1][1] and self.board[1][1] == self.board[2][0]:
            return True, self.board[2][0]

        if self.moves_made == 9:
            return True, None

        return False, None
        

    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Players.empty:
                    moves.append((i, j))
        return moves

    def move(self, move):
        self.board[move[0]][move[1]] = self.turn
        self.moves_made += 1
        self.switch_turns()

    def undo_move(self, move):
        self.board[move[0]][move[1]] = Players.empty
        self.moves_made -= 1
        self.switch_turns()

class ValueNode:
    def __init__(self, value):
        self.children = {}
        self.value = value

class ValueTree:
    def __init__(self):
        self.root = ValueNode(0.5)
        self.curr = self.root

    def reset(self):
        self.curr = self.root

    def move_and_update(self, our_move, their_move, lr):
        pass

class Agent:
    def __init__(self):
        self.state = State()
        self.tree = ValueTree()

        self.last_move = (None, False)

        self.lr = 1
        self.lr_mult = 0.99999

        self.exploration_rate = 0.5
        self.exploration_rate_mult = 0.99999

    def reset(self):
        self.state = State()
        self.tree.reset()

    def get_move(self):
        
        explore = bool(random.randint(0, 1))
        available_moves = self.state.available_moves()
        if explore:
            self.last_move = (random.choice(available_moves), True)
            return self.last_move[0]
        else:
            best_move = None
            best_value = -1
            curr_children = self.tree.curr.children
            for move in available_moves:
                if move in curr_children:
                    value = curr_children[move].value
                    if value > best_value:
                        best_value = value
                        best_move = move
                else:
                    self.state.move(move)
                    value = 0.5
                    if self.state.is_terminal()[0]:
                        value = 1
                    newnode = ValueNode(value)
                    curr_children[move] = newnode

                    if value > best_value:
                        best_value = value
                        best_move = move

                    self.state.undo_move(move)

            self.last_move = (best_move, False)
            return best_move

    def opponent_move(self, move):
        self.tree.move_and_update(self.last_move, move, self.lr)
        self.state.move(self.last_move)
        self.state.move(move)


    



    