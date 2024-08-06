import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from copy import deepcopy
import pickle
from tqdm import tqdm

from UT3 import *


class Player:
    def __init__(self, symbol):
        self.symbol = symbol

    def make_move(self, game):
        raise NotImplementedError("Subclass must implement abstract method")

    def game_over(self, winner):
        pass

    def opponent_move(self, move):
        pass


class RandomPlayer(Player):
    def make_move(self, game):
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves)

    def game_over(self, winner):
        print(f"Game over. Winner: {winner}")

    def opponent_move(self, move):
        print(f"Opponent moved: {move}")


class HumanPlayer(Player):
    def make_move(self, game):
        valid_moves = game.get_valid_moves()
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
        self.game_state = deepcopy(game_state)
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

    def make_move(self, game):
        # valid_moves = game.get_valid_moves()
        # root = MCTSNode(self.game_state)
        root = MCTSNode(game_state=game)

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

    def get_state_key(self, game): # turns a game object of U3 into a state representation - str(dict( U3 fields ))
        # return str(game.board)
        tmp = deepcopy(game)
        # state = str({'board':tmp.board, 'current_player':tmp.current_player, 'last_move':tmp.last_move,
        #              'game_over':tmp.game_over, 'winner':tmp.winner, 'moves':tmp.moves,
        #              'free_move_from_last_move':tmp.free_move_from_last_move,
        #              'place_holder':tmp.place_holder, 'meta_board':tmp.meta_board})
        state = str({'board': tmp.board, 'last_move': tmp.last_move,})
        return state

    def state_key_to_u3(self, state_key):
        # Parse the string representation of the dictionary
        state_dict = eval(state_key)
        # Create a new U3 object
        new_game = U3()

        # Set the fields from the state dictionary
        new_game.board = state_dict['board']
        # new_game.current_player = state_dict['current_player']
        new_game.last_move = state_dict['last_move']
        # new_game.game_over = state_dict['game_over']
        # new_game.winner = state_dict['winner']
        # new_game.moves = state_dict['moves']
        # new_game.free_move_from_last_move = state_dict['free_move_from_last_move']
        # new_game.place_holder = state_dict['place_holder']
        # new_game.meta_board = state_dict['meta_board']

        return new_game

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {str(action): 0.0 for action in self.get_possible_actions(state)}
            # self.q_table[state] = {str(action): 0.0 for action in self.get_possible_actions(state)}
        return self.q_table[state].get(str(action), 0.0)

    def get_possible_actions(self, state):  # ORIGINAL
        game = self.state_key_to_u3(state)
        valid_moves = game.get_valid_moves()
        return valid_moves

    # def get_possible_actions(self, game):
    #     return game.get_valid_moves()

    def choose_action(self, game, training=False):
        state = self.get_state_key(game)
        if training and np.random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
            # return random.choice(self.get_possible_actions(game))
        else:
            return max(self.get_possible_actions(state), key=lambda a: self.get_q_value(state, a))
            # return max(self.get_possible_actions(game), key=lambda a: self.get_q_value(state, a))

    def make_move(self, game):
        return self.choose_action(game, training=False)

    def update_q_value(self, state, action, next_state, reward):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.get_possible_actions(next_state))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][str(action)] = new_q

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


class DQNNet(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQNNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent(Player):
    def __init__(self, symbol, input_channels=3, action_size=81, memory_size=10000, batch_size=64, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, training=False):
        super().__init__(symbol)
        self.input_channels = input_channels
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQNNet(input_channels, action_size).to(self.device)
        self.target_model = DQNNet(input_channels, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def get_state(self, game):
        state = np.zeros((self.input_channels, 9, 9), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        cell = game.board[i][j][k][l]
                        if cell == self.symbol:
                            state[0, i * 3 + k, j * 3 + l] = 1
                        elif cell != ' ' and cell != '~':
                            state[1, i * 3 + k, j * 3 + l] = 1
                        elif cell == '~':
                            state[2, i * 3 + k, j * 3 + l] = 1

        # Add current player and free move information to the state
        if game.current_player == self.symbol:
            state[0, :, :] += 0.1  # Slightly increase values in the player's channel
        else:
            state[1, :, :] += 0.1  # Slightly increase values in the opponent's channel

        if game.free_move_from_last_move:
            state[2, :, :] += 0.1  # Slightly increase values in the '~' channel to indicate free move

        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def make_move(self, game, training=False):
        state = self.get_state(game)
        if np.random.rand() <= self.epsilon and training:
            return random.choice(game.get_valid_moves())

        act_values = self.model(state)
        valid_moves = game.get_valid_moves()
        valid_indices = [self.move_to_index(move) for move in valid_moves]
        best_move_index = torch.argmax(act_values[0, valid_indices]).item()
        return valid_moves[best_move_index]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def move_to_index(move):
        br, bc, sr, sc = move
        return br * 27 + bc * 9 + sr * 3 + sc

    @staticmethod
    def index_to_move(index):
        br = index // 27
        bc = (index % 27) // 9
        sr = (index % 9) // 3
        sc = index % 3
        return (br, bc, sr, sc)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def train_dqn_agent(agent, opponent, num_episodes):
    progress_bar = tqdm(range(num_episodes), desc="Training", unit="episode")
    for episode in progress_bar:
        game = U3()
        state = agent.get_state(game)
        done = False

        while not done:
            if game.current_player == agent.symbol:
                move = agent.make_move(game, training=True)
                game.make_move(*move)
                next_state = agent.get_state(game)
                reward = 1 if game.winner == agent.symbol else -1 if game.winner else 0
                done = game.game_over
                agent.remember(state, agent.move_to_index(move), reward, next_state, done)
                state = next_state
                agent.train()
            else:
                opponent_move = opponent.make_move(game)
                game.make_move(*opponent_move)
                done = game.game_over

        if episode % 100 == 0:
            print(f"Episode: {episode}, Epsilon: {agent.epsilon:.2f}")
            agent.update_target_model()

    return agent





class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape=(3, 9, 9), action_size=81):
        super(AlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * input_shape[1] * input_shape[2], 1024)
        self.fc_value = nn.Linear(1024, 1)
        self.fc_policy = nn.Linear(1024, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.fc_value(x))
        policy = F.softmax(self.fc_policy(x), dim=1)
        return policy, value

class MCTSNodeAZ:
    def __init__(self, game, parent=None, action=None, prior=0, cpuct=1.0):
        self.game = game
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.cpuct = cpuct

    def expand(self, action_priors):
        for action, prob in enumerate(action_priors):
            if prob > 0:
                child_game = deepcopy(self.game)
                move = AlphaZeroPlayer.index_to_move(action)
                if move in child_game.get_valid_moves():
                    child_game.make_move(*move)
                    self.children[action] = MCTSNodeAZ(child_game, self, action, prob, self.cpuct)

    def select(self):
        return max(self.children.items(), key=lambda item: item[1].get_ucb(self.visit_count))

    def get_ucb(self, parent_visit_count):
        if self.visit_count == 0:
            return float('inf')
        return (self.value_sum / self.visit_count) + \
               (self.cpuct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count))



class AlphaZeroPlayer(Player):
    def __init__(self, symbol, input_shape=(3, 9, 9), action_size=81, mcts_simulations=800, cpuct=1.0):
        self.symbol = symbol
        self.mcts_simulations = mcts_simulations
        self.cpuct = cpuct
        self.action_size = action_size
        self.net = AlphaZeroNet(input_shape, action_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def get_state_tensor(self, game):
        tensor = np.zeros((3, 9, 9), dtype=np.float32)
        for br in range(3):
            for bc in range(3):
                for sr in range(3):
                    for sc in range(3):
                        if game.board[br][bc][sr][sc] == self.symbol:
                            tensor[0, br * 3 + sr, bc * 3 + sc] = 1
                        elif game.board[br][bc][sr][sc] != ' ':
                            tensor[1, br * 3 + sr, bc * 3 + sc] = 1

        valid_moves = game.get_valid_moves()
        for move in valid_moves:
            br, bc, sr, sc = move
            tensor[2, br * 3 + sr, bc * 3 + sc] = 1

        return torch.FloatTensor(tensor).unsqueeze(0)

    def get_valid_moves_mask(self, game):
        mask = np.zeros(self.action_size, dtype=np.float32)
        valid_moves = game.get_valid_moves()
        for move in valid_moves:
            index = self.move_to_index(move)
            mask[index] = 1
        return mask

    @staticmethod
    def move_to_index(move):
        br, bc, sr, sc = move
        return br * 27 + bc * 9 + sr * 3 + sc

    @staticmethod
    def index_to_move(index):
        br = index // 27
        bc = (index % 27) // 9
        sr = (index % 9) // 3
        sc = index % 3
        return (br, bc, sr, sc)

    def mcts_search(self, game):
        root = MCTSNodeAZ(game, cpuct=self.cpuct)

        for _ in range(self.mcts_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.children and not node.game.game_over:
                action, node = node.select()
                search_path.append(node)

            # Expansion and evaluation
            state = self.get_state_tensor(node.game)
            action_probs, value = self.net(state)
            valid_moves = self.get_valid_moves_mask(node.game)
            action_probs = action_probs.detach().numpy().flatten() * valid_moves
            action_probs /= np.sum(action_probs) + 1e-8

            if not node.game.game_over:
                node.expand(action_probs)

            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value.item() if node.game.current_player == self.symbol else -value.item()
                node.visit_count += 1

        # Return normalized visit counts as action probabilities
        actions_visits = [(action, child.visit_count) for action, child in root.children.items()]
        if not actions_visits:
            return np.ones(self.action_size) / self.action_size  # Uniform distribution if no valid moves
        actions, visits = zip(*actions_visits)
        probs = np.zeros(self.action_size)
        probs[list(actions)] = np.array(visits) / np.sum(visits)
        return probs

    def make_move(self, game):
        probs = self.mcts_search(game)
        action = np.random.choice(len(probs), p=probs)
        return self.index_to_move(action)

    def train(self, memory):
        batch = random.sample(memory, min(len(memory), 32))
        state_batch = torch.cat([data[0] for data in batch])
        mcts_probs_batch = torch.from_numpy(np.array([data[1] for data in batch])).float()
        value_batch = torch.from_numpy(np.array([data[2] for data in batch])).float()

        self.optimizer.zero_grad()
        policy_batch, value_batch_pred = self.net(state_batch)

        value_loss = F.mse_loss(value_batch_pred.view(-1), value_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * torch.log(policy_batch + 1e-8), 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()


class MinimaxPlayer(Player):
    def __init__(self, symbol, depth=3):
        super().__init__(symbol)
        self.depth = depth  # Depth to which the minimax algorithm searches
        self.nodes = 0

    def make_move(self, game):
        _, move = self.minimax(game, self.depth, True, float('-inf'), float('inf'))
        return move

    def minimax(self, game, depth, maximizing_player, alpha, beta):
        if depth == 0 or game.game_over:
            return self.evaluate(game), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in game.get_valid_moves():
                self.nodes += 1
                game_copy = deepcopy(game)
                game_copy.make_move(*move)
                eval, _ = self.minimax(game_copy, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:

                    break  # Beta cut-off
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in game.get_valid_moves():
                self.nodes += 1
                game_copy = deepcopy(game)
                game_copy.make_move(*move)
                eval, _ = self.minimax(game_copy, depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_move

    def evaluate(self, game):
        if game.winner == self.symbol:
            return 100  # Favorable outcome for MinimaxPlayer
        elif game.winner and game.winner != self.symbol:
            return -100  # Unfavorable outcome for MinimaxPlayer
        else:
            return 0  # Neutral outcome (e.g., draw or game not finished)

    def set_game_state(self, game_state):
        self.game_state = deepcopy(game_state)


