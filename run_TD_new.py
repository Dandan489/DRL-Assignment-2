from os import stat
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import os
from student_agent import Game2048Env
import matplotlib.pyplot as plt

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
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            # for syms_ in syms:
            self.symmetry_patterns.append(syms)

        # print(self.symmetry_patterns)

    def generate_symmetries(self, pattern):
        return [
            pattern,
            rotate_90(pattern),
            rotate_90(rotate_90(pattern)),
            rotate_90(rotate_90(rotate_90(pattern))),
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
        for i, pattern in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                total += self.weights[i][feature]
        return total

    def update(self, board, delta, alpha):
        for i, pattern in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                self.weights[i][feature] += (alpha * delta / (8 * len(patterns)))

    def evaluate(self, env, action, approximator):
        """Evaluate function: computes the afterstate and returns its estimated value."""
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
        return reward + approximator.value(sim_env.board)

def save_weights(approximator, filename):
    with open(filename, "wb") as f:
        pickle.dump(approximator.weights, f)

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []

    epsilon_start = 0.1
    epsilon_decay = 0.999
    epsilon = epsilon_start
    
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            values = []
            after_state = []
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
                reward = sim_env.score - previous_score
                values.append(reward + gamma * approximator.value(sim_env.board))
                after_state.append(sim_env.board)
            action = legal_moves[np.argmax(values)]
            after_state = after_state[np.argmax(values)]

            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            a_next = max(legal_moves, key=lambda a: approximator.evaluate(env, a, approximator))
            sim_env = copy.deepcopy(env)
            if a_next == 0:
                sim_env.move_up()
            elif a_next == 1:
                sim_env.move_down()
            elif a_next == 2:
                sim_env.move_left()
            elif a_next == 3:
                sim_env.move_right()
            s_next, r_next = sim_env.board, sim_env.score - env.score
            trajectory.append((r_next, s_next.copy(), after_state.copy()))
            state = next_state

        for reward, state_next, after in reversed(trajectory):
            delta = reward + approximator.value(state_next) - approximator.value(after)
            approximator.update(after, delta, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        if (episode + 1) % 1000 == 0:
            weight_filename = os.path.join(weights_dir, f"weights_{episode+1}_2.pkl")
            save_weights(approximator, weight_filename)
            print(f"Saved weights to {weight_filename}")

    return final_scores

def train(approx):
    env = Game2048Env()
    final_scores = td_learning(env, approx, num_episodes=10000, alpha=0.16, gamma=0.99, epsilon=0.1)

def load_weights(approx, path):
    with open(path, "rb") as f:
        approx.weights = pickle.load(f)

def run_game(approx):
    env = Game2048Env()
    scores = []
    
    for _ in range(10):
        done = False
        state = env.reset()
        highest = 0

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            
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
                values.append(approx.value(sim_env.board))
            best_action = legal_moves[np.argmax(values)]

            action = best_action
            state, reward, done, _ = env.step(action)
            highest = max(highest, np.max(env.board))

        print(f"Score: {env.score}, Highest: {highest}")
        scores.append(env.score)
        
    print(np.mean(scores))
        
if __name__ == "__main__":
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
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    load_weights(approximator, "weights/weights_10000.pkl")
    # train(approximator)
    run_game(approximator)
