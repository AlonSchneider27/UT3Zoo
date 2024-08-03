import random
import math
from copy import deepcopy


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