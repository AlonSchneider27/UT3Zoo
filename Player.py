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


def play_game(player1, player2, game):
    players = [player1, player2]
    current_player_index = 0
    turn = 1

    while not game.game_over:
        current_player = players[current_player_index]
        valid_moves = game.get_valid_moves()
        move = current_player.make_move(valid_moves)

        print(f"Turn: {turn},Player {current_player.symbol} moves: {move}")
        game.make_move(*move)
        game.print_board(place_holder=True)
        print("\n")  # Add a blank line for readability

        # Inform the other player about the move
        players[1 - current_player_index].opponent_move(game.last_move)

        # Switch players
        current_player_index = 1 - current_player_index

        turn += 1

    # Game over
    player1.game_over(game.winner)
    player2.game_over(game.winner)
    print(f"Game over. Winner: {game.winner}")


# Now let's run a game
if __name__ == "__main__":
    game = U3()
    player1 = RandomPlayer('X')
    player2 = RandomPlayer('O')

    print("Starting the game!")
    game.print_board()
    play_game(player1, player2, game)