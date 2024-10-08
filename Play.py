from Player import *
from UT3 import *


def play_game(player1, player2, game):
    players = [player1, player2]
    current_player_index = 0
    turn = 1

    while not game.game_over:
        current_player = players[current_player_index]
        valid_moves = game.get_valid_moves()

        if isinstance(current_player, (MCTSPlayer, QLearningPlayer, MinimaxPlayer)):
            current_player.set_game_state(game)
        # DQNAgent doesn't need set_game_state, it uses get_state in make_move

        if isinstance(current_player, HumanPlayer):
            print(f"\nTurn {turn}: It's your turn, Player {current_player.symbol}")
            game.print_board(place_holder=True)
        else:
            print(f"\nTurn {turn}: Player {current_player.symbol}'s turn")

        move = current_player.make_move(game)

        print(f"Player {current_player.symbol} moves: {move}")
        game.make_move(*move)

        if not isinstance(current_player, HumanPlayer):
            game.print_board(place_holder=True)

        print("\n")  # Add a blank line for readability

        # Inform the other player about the move
        if hasattr(players[1 - current_player_index], 'opponent_move'):
            players[1 - current_player_index].opponent_move(move)

        # Switch players
        current_player_index = 1 - current_player_index

        turn += 1

    # Game over
    print(f"Game over. Winner: {game.winner}")


def Zoo():
    while True:
        game = U3()

        print("\nChoose player types:")
        print("1. Human")
        print("2. Random AI")
        print("3. MCTS AI")
        print("4. Q-Learning AI")
        print("5. DQN AI")
        print("6. AlphaZero AI")
        print("7. Minimax AI")

        player1_type = input("Select Player 1 type (1, 2, 3, 4, 5, 6, or 7): ")
        player2_type = input("Select Player 2 type (1, 2, 3, 4, 5, 6, or 7): ")

        def create_player(player_type, symbol):
            if player_type == '1':
                return HumanPlayer(symbol)
            elif player_type == '2':
                return RandomPlayer(symbol)
            elif player_type == '3':
                return MCTSPlayer(symbol)
            elif player_type == '4':
                q_player = QLearningPlayer(symbol)
                q_player.load_q_table('q_player_best_QR5000_250MCTS.pkl')  # Load the trained Q-table
                return q_player
            elif player_type == '5':
                dqn_player = DQNAgent(symbol)
                dqn_player.load_model('dqn_u3_model.pth')
                return dqn_player
            elif player_type == '6':
                alpha_zero_player = AlphaZeroPlayer(symbol)
                alpha_zero_player.load_model('ZERO_model_iteration_9.pth')
                return alpha_zero_player
            elif player_type == '7':
                return MinimaxPlayer(symbol)
            else:
                raise ValueError("Invalid player type")

        player1 = create_player(player1_type, 'X')
        player2 = create_player(player2_type, 'O')

        print("\nStarting the game!")
        game.print_board(place_holder=True)
        play_game(player1, player2, game)
        print(f'Final Board after {len(game.moves)} Turns:')
        game.print_board(place_holder=True)
        print('Final Meta Board:')
        game.print_meta_board()

        play_again = int(input("\nDo you want to play another game? YES<-1 No<-0 "))
        if play_again != 1:
            break

    print("Thanks for playing!")

    print("Thanks for playing!")
if __name__ == "__main__":
    Zoo()