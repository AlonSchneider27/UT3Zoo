from Player import *
from UT3 import *

# RANDOM vs RANDOM
# def play_game(player1, player2, game):
#     players = [player1, player2]
#     current_player_index = 0
#     turn = 1
#
#     while not game.game_over:
#         current_player = players[current_player_index]
#         valid_moves = game.get_valid_moves()
#         move = current_player.make_move(valid_moves)
#
#         print(f"Turn: {turn},Player {current_player.symbol} moves: {move}")
#         game.make_move(*move)
#         game.print_board(place_holder=True)
#         print("\n")  # Add a blank line for readability
#
#         # Inform the other player about the move
#         players[1 - current_player_index].opponent_move(game.last_move)
#
#         # Switch players
#         current_player_index = 1 - current_player_index
#
#         turn += 1
#     print(f"Game over. Winner: {game.winner}")

## HUMAN VS RANDOM
def play_game(player1, player2, game):
    players = [player1, player2]
    current_player_index = 0
    turn = 1

    while not game.game_over:
        current_player = players[current_player_index]
        valid_moves = game.get_valid_moves()

        if isinstance(current_player, HumanPlayer):
            print(f"\nTurn {turn}: It's your turn, Player {current_player.symbol}")
            game.print_board(place_holder=True)
        else:
            print(f"\nTurn {turn}: Player {current_player.symbol}'s turn")

        move = current_player.make_move(valid_moves)

        print(f"Player {current_player.symbol} moves: {move}")
        game.make_move(*move)

        if not isinstance(current_player, HumanPlayer):
            game.print_board(place_holder=True)

        print("\n")  # Add a blank line for readability

        # Inform the other player about the move
        players[1 - current_player_index].opponent_move(game.last_move)

        # Switch players
        current_player_index = 1 - current_player_index

        turn += 1

    # Game over
    print(f"Game over. Winner: {game.winner}")


# Now let's run a game
if __name__ == "__main__":

    ## RANDOM VS RANDOM
    # game = U3()
    # player1 = RandomPlayer('X')
    # player2 = RandomPlayer('O')
    #
    # print("Starting the game!")
    # game.print_board()
    # play_game(player1, player2, game)
    # print('Final Board:')
    # game.print_meta_board()

    ## HUMAN VS RANDOM
    game = U3()

    print("Choose player types:")
    print("1. Human")
    print("2. Random AI")

    player1_type = input("Select Player 1 type (1 or 2): ")
    player2_type = input("Select Player 2 type (1 or 2): ")

    player1 = HumanPlayer('X') if player1_type == '1' else RandomPlayer('X')
    player2 = HumanPlayer('O') if player2_type == '1' else RandomPlayer('O')

    print("\nStarting the game!")
    game.print_board(place_holder=True)
    play_game(player1, player2, game)
    print('Final Board:')
    game.print_board(place_holder=True)
    print('Final Meta Board:')
    game.print_meta_board()