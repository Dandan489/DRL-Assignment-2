# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict
import json

def rotate_90(positions):
    return [(y, 3 - x) for (x, y) in positions]

def reflect(positions):
    return [(x, 3 - y) for (x, y) in positions]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        # self.V_init = 0.0
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.E = 0.0
        self.A = 0.0
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        return [
            pattern,
            rotate_90(pattern),
            rotate_90(rotate_90(pattern)),
            rotate_90(rotate_90(rotate_90(pattern))),
            reflect(pattern),
            rotate_90(reflect(pattern)),
            rotate_90(rotate_90(reflect(pattern))),
            rotate_90(rotate_90(rotate_90(reflect(pattern))))
        ]

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        total = 0
        for i, _ in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                total += self.weights[i][feature]
        return total

    def update(self, board, delta, t):
        # if(self.A != 0):
        #     alpha = np.abs(self.E) / self.A
        # else:
        #     alpha = 1
        
        alpha = 0.32
        for i, _ in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                self.weights[i][feature] += (alpha * delta / (8 * len(patterns)) * 1)
                    
        # self.E += delta
        # self.A += np.abs(delta)

    def evaluate(self, env):
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            return 0
        
        values = []
        actions = []
        for action in legal_moves:
            sim_env = copy.deepcopy(env)
            if action == 0:
                sim_env.move_up()
            elif action == 1:
                sim_env.move_down()
            elif action == 2:
                sim_env.move_left()
            elif action == 3:
                sim_env.move_right()
            reward = sim_env.score - env.score
            actions.append(action)
            values.append(reward + self.value(sim_env.board))
            
        return actions[np.argmax(values)]

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

# def load_weights(approx, path):
#     print(f"Loading {path}")
#     with open(path, "rb") as f:
#         approx.weights = pickle.load(f)

def load_weights(approx, path):
    with open(path, "r") as f:
        raw = json.load(f)
    weights = [
        defaultdict(float, {tuple(map(int, k.split(","))): v for k, v in d.items()})
        for d in raw
    ]
    approx.weights = weights

loaded = None
approximator_0 = None

def get_action(state, score):
    patterns = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
    ]
    
    global loaded
    global approximator_0
    if(loaded is None):
        approximator_0 = NTupleApproximator(board_size=4, patterns=patterns)
        load_weights(approximator_0, "weights.json")
        loaded = 1
    
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]

    values = []
    for action in legal_moves:
        sim_env = copy.deepcopy(env)
        if action == 0:
            sim_env.move_up()
        elif action == 1:
            sim_env.move_down()
        elif action == 2:
            sim_env.move_left()
        elif action == 3:
            sim_env.move_right()
        reward = sim_env.score - env.score
        values.append(reward + approximator_0.value(sim_env.board))
    best_action = legal_moves[np.argmax(values)]
    
    return best_action


