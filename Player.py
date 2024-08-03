import random
import math
from copy import deepcopy
import pickle
from UT3 import *
import numpy as np


class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    def make_move(self, valid_moves):
        raise NotImplementedError("Subclass must implement abstract method")

    def game_over(self, winner):
        pass

    def opponent_move(self, move):
        pass


class RandomPlayer(Player):
    def make_move(self, valid_moves):
        return random.choice(valid_moves)

    def game_over(self, winner):
        print(f"Game over. Winner: {winner}")

    def opponent_move(self, move):
        print(f"Opponent moved: {move}")


class HumanPlayer(Player):
    def make_move(self, valid_moves):
        while True:
            try:
                # Input is in the form of a 4-digit number, e.g. "0210"
                move = input("Enter your move as a 4-digit number (big_row big_col small_row small_col): ")
                if len(move) != 4 or not move.isdigit():
                    raise ValueError("Input must be a 4-digit number")
                big_row = int(move[0])
                big_col = int(move[1])
                small_row = int(move[2])
                small_col = int(move[3])
                if (big_row, big_col, small_row, small_col) in valid_moves:
                    return big_row, big_col, small_row, small_col
                else:
                    print("Invalid move. Try again.")
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter a 4-digit number.")

    def opponent_move(self, move):
        print(f"Opponent moved: {move}")


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.score = 0


class MCTSPlayer(Player):
    def __init__(self, symbol, iterations=1000, exploration_weight=1.41):
        super().__init__(symbol)
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def make_move(self, valid_moves):
        root = MCTSNode(self.game_state)

        for _ in range(self.iterations):
            node = self.select(root)
            simulation_result = self.simulate(node)
            self.backpropagate(node, simulation_result)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def select(self, node):
        while not node.game_state.game_over:
            if len(node.children) < len(node.game_state.get_valid_moves()):
                return self.expand(node)
            else:
                node = self.uct_select(node)
        return node

    def expand(self, node):
        valid_moves = node.game_state.get_valid_moves()
        untried_moves = [move for move in valid_moves if not any(child.move == move for child in node.children)]
        move = random.choice(untried_moves)
        new_state = deepcopy(node.game_state)
        new_state.make_move(*move)
        child_node = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def simulate(self, node):
        state = deepcopy(node.game_state)
        while not state.game_over:
            valid_moves = state.get_valid_moves()
            move = random.choice(valid_moves)
            state.make_move(*move)
        return 1 if state.winner == self.symbol else 0 if state.winner is None else -1

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.score += result
            node = node.parent

    def uct_select(self, node):
        return max(node.children, key=lambda c: c.score / c.visits +
                                                self.exploration_weight * math.sqrt(math.log(node.visits) / c.visits))

    def opponent_move(self, move):
        # self.game_state.make_move(*move)
        pass
    def set_game_state(self, game_state):
        self.game_state = deepcopy(game_state)


class QLearningPlayer(Player):
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        super().__init__(symbol)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state_key(self, game):
        return str(game.board)

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {str(action): 0.0 for action in self.get_possible_actions(state)}
        return self.q_table[state].get(str(action), 0.0)

    def get_possible_actions(self, state):
        game = U3()
        game.board = eval(state)
        return game.get_valid_moves()

    def choose_action(self, game, training=False):
        state = self.get_state_key(game)
        if training and np.random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            return max(self.get_possible_actions(state), key=lambda a: self.get_q_value(state, a))

    # def make_move(self, valid_moves):
    #     return self.choose_action(self.game_state, training=False)

    def update_q_value(self, state, action, next_state, reward):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.get_possible_actions(next_state))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][str(action)] = new_q

    def train(self, num_episodes, opponent):
        for episode in range(num_episodes):
            game = U3()
            state = self.get_state_key(game)
            while not game.game_over:
                action = self.choose_action(game, training=True)
                game.make_move(*action)
                next_state = self.get_state_key(game)
                reward = self.get_reward(game)
                self.update_q_value(state, action, next_state, reward)
                state = next_state
                if not game.game_over:
                    if isinstance(opponent, QLearningPlayer):  # if the opponent is also QLearningPlayer
                        opponent_move = opponent.choose_action(game, training=True)
                    else:
                        opponent_move = opponent.make_move(game.get_valid_moves())
                    game.make_move(*opponent_move)
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")

    def get_reward(self, game):
        if game.game_over:
            if game.winner == self.symbol:
                return 1
            elif game.winner is None:
                return 0
            else:
                return -1
        return 0

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

    def set_game_state(self, game_state):
        self.game_state = deepcopy(game_state)


