import random
from UT3 import U3

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
