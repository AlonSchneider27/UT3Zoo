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


class HumanPlayer(Player):
    def make_move(self, valid_moves):
        while True:
            try:
                move = input("Enter your move (big_row, big_col, small_row, small_col): ")
                big_row, big_col, small_row, small_col = map(int, move.split(','))
                if (big_row, big_col, small_row, small_col) in valid_moves:
                    return big_row, big_col, small_row, small_col
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Please enter four integers separated by commas.")

    def opponent_move(self, move):
        print(f"Opponent moved: {move}")
