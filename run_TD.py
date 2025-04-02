from os import stat
import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import os
from student_agent import Game2048Env
import time

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
        for i, _ in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                total += self.weights[i][feature]
        return total

    def update(self, board, delta, alpha):
        for i, _ in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                self.weights[i][feature] += (alpha * delta / (8 * len(patterns)))

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

def save_weights(approximator, filename):
    with open(filename, "wb") as f:
        pickle.dump(approximator.weights, f)

def td_learning(env, approximator_0, approximator_2048, num_episodes=50000, alpha_0=0.01, alpha_2048=0.16, gamma=0.99, epsilon=0.1):
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
    
    weights_0_dir = "weights"
    weights_2048_dir = "weights_2048"
    os.makedirs(weights_0_dir, exist_ok=True)
    os.makedirs(weights_2048_dir, exist_ok=True)

    for episode in range(num_episodes):
        
        # prev_time = time.time()
        # now_time = time.time()
        
        state = env.reset()
        trajectory = []
        # previous_score = 0
        done = False
        # max_tile = np.max(state)
        
        prev_state = env.board
        a_next = 0
        # highest = 0
        approximator = approximator_0
        tag = 0
        
        # step = 0
        
        while not done:
            # legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            # if not legal_moves:
            #     break
            # values = []
            # after_state = []
            # for action in legal_moves:
            #     sim_env = copy.deepcopy(env)
            #     if action == 0:
            #         sim_env.move_up()
            #     elif action == 1:
            #         sim_env.move_down()
            #     elif action == 2:
            #         sim_env.move_left()
            #     elif action == 3:
            #         sim_env.move_right()
            #     reward = sim_env.score - previous_score
            #     values.append(reward + gamma * approximator.value(sim_env.board))
            #     after_state.append(sim_env.board)
            # action = legal_moves[np.argmax(values)]
            # after_state = after_state[np.argmax(values)]

            
            
            next_state, new_score, done, _ = env.step(a_next)
            # incremental_reward = new_score - previous_score
            # previous_score = new_score
            # max_tile = max(max_tile, np.max(next_state))
            
            # now_time = time.time()
            # print(f"Step time: {now_time - prev_time:.8f} seconds")
            # prev_time = now_time

            a_next = approximator.evaluate(env)
            
            # now_time = time.time()
            # print(f"Eval time: {now_time - prev_time:.8f} seconds")
            # prev_time = now_time
            
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
            trajectory.append((r_next, s_next.copy(), prev_state.copy(), tag))
            
            # now_time = time.time()
            # print(f"Traj time: {now_time - prev_time:.8f} seconds")
            # prev_time = now_time
            
            state = next_state
            prev_state = s_next
            
            if(r_next >= 2048):
                approximator = approximator_2048
                tag = 1
        
        # now_time = time.time()
        # print(f"Walk time: {now_time - prev_time:.4f} seconds")
        # prev_time = now_time
                
        for reward, state_next, after, tra_tag in reversed(trajectory):
            if(tra_tag == 0):
                delta = reward + approximator_0.value(state_next) - approximator_0.value(after)
                approximator_0.update(after, delta, alpha_0)
            else:
                delta = reward + approximator_2048.value(state_next) - approximator_2048.value(after)
                approximator_2048.update(after, delta, alpha_2048)
                
        # now_time = time.time()
        # print(f"Update time: {now_time - prev_time:.4f} seconds")
        # prev_time = now_time

        final_scores.append(env.score)
        success_flags.append(1 if tag == 1 else 0)
        
        # print(episode, env.score, step)
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
            
        if (episode + 1) % 1000 == 0:
            weight_filename = os.path.join(weights_0_dir, f"weights_{episode+1}_4.pkl")
            save_weights(approximator_0, weight_filename)
            print(f"Saved weights to {weight_filename}")
            
            weight_filename = os.path.join(weights_2048_dir, f"weights_{episode+1}_4.pkl")
            save_weights(approximator_2048, weight_filename)
            print(f"Saved weights to {weight_filename}")

    return final_scores

def train(approx_0, approx_2048):
    env = Game2048Env()
    final_scores = td_learning(env, approx_0, approx_2048, num_episodes=10000, alpha_0=0.16, alpha_2048=0.32, gamma=0.99, epsilon=0.1)

def load_weights(approx, path):
    print(f"Loading {path}")
    with open(path, "rb") as f:
        approx.weights = pickle.load(f)

def run_game(approx):
    env = Game2048Env()
    scores = []
    
    for _ in range(20):
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
                reward = sim_env.score - env.score
                values.append(reward + approx.value(sim_env.board))
            best_action = legal_moves[np.argmax(values)]

            action = best_action
            _, reward, done, _ = env.step(action)
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
    approximator_0 = NTupleApproximator(board_size=4, patterns=patterns)
    approximator_2048 = NTupleApproximator(board_size=4, patterns=patterns)
    weight_0_path = "weights/weights_10000_3.pkl"
    weight_2048_path = "weights_2048/weights_10000_3.pkl"
    load_weights(approximator_0, weight_0_path)
    load_weights(approximator_2048, weight_2048_path)
    train(approximator_0, approximator_2048)
    # run_game(approximator_0)
