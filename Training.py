from Play import *
from Player import *
from UT3 import *
from tqdm import tqdm

def train_q_player(q_player, opponent, num_episodes):
    for episode in tqdm(range(num_episodes), desc=f"Training with {type(opponent).__name__}", unit="episode"):
        game = U3()
        state = q_player.get_state_key(game)
        while not game.game_over:
            action = q_player.choose_action(game, training=True)
            game.make_move(*action)
            if game.game_over: break
            next_state = q_player.get_state_key(game)
            reward = q_player.get_reward(game)
            q_player.update_q_value(state, action, next_state, reward)
            state = next_state
            if not game.game_over:
                if isinstance(opponent, QLearningPlayer):
                    opponent_move = opponent.choose_action(game, training=True)
                else:
                    opponent_move = opponent.make_move(game)
                game.make_move(*opponent_move)
        # if (episode + 1) % 100 == 0:
        #     print(f"Episode {episode + 1}/{num_episodes} completed")
    return q_player


def train_q_player_multiple_opponents(q_player, opponents_and_episodes):
    for opponent, num_episodes in opponents_and_episodes:
        print(f"Training against {opponent.__class__.__name__}...")
        train_q_player(q_player, opponent, num_episodes)
    return q_player


def evaluate_players(player1, player2, num_games=100):
    wins = {player1.symbol: 0, player2.symbol: 0, 'Draw': 0}
    total_moves = 0
    games = []

    for game_num in range(num_games):
        game = U3()
        moves = 0
        while not game.game_over:
            current_player = player1 if game.current_player == player1.symbol else player2
            if isinstance(current_player, (MCTSPlayer, QLearningPlayer)):
                current_player.set_game_state(game)
            move = current_player.make_move(game)
            game.make_move(*move)
            moves += 1

        total_moves += moves
        games.append((game.winner, moves))

        if game.winner:
            wins[game.winner] += 1
        else:
            wins['Draw'] += 1

        if (game_num + 1) % 10 == 0:
            print(f"Completed {game_num + 1} games...")

    avg_moves = total_moves / num_games
    win_rate_player1 = (wins[player1.symbol] / num_games) * 100
    win_rate_player2 = (wins[player2.symbol] / num_games) * 100
    draw_rate = (wins['Draw'] / num_games) * 100

    print(f"\nEvaluation Results ({player1.__class__.__name__} vs {player2.__class__.__name__}):")
    print(f"Total Games: {num_games}")
    print(f"{player1.__class__.__name__} ({player1.symbol}) Wins: {wins[player1.symbol]} ({win_rate_player1:.2f}%)")
    print(f"{player2.__class__.__name__} ({player2.symbol}) Wins: {wins[player2.symbol]} ({win_rate_player2:.2f}%)")
    print(f"Draws: {wins['Draw']} ({draw_rate:.2f}%)")
    print(f"Average Moves per Game: {avg_moves:.2f}")

    return wins, games
# Usage example:
if __name__ == '__main__':
    """
    QLearningPlayer Training
    """
    #
    # game = U3()
    # q_player = QLearningPlayer('X')
    #
    # opponents_and_episodes = [
    #     (RandomPlayer('O'), 500),
    #     (QLearningPlayer('O'), 500),
    #     (MCTSPlayer('O'), 1)
    # ]
    #
    # trained_q_player = train_q_player_multiple_opponents(q_player, opponents_and_episodes)
    # trained_q_player.save_q_table('q_player_best.pkl')
    #
    # print("\nEvaluating Q-Learning vs Random:")
    # q_player = QLearningPlayer('X')
    # q_player.load_q_table('q_player_best.pkl')
    # random_player = RandomPlayer('O')
    # wins, games = evaluate_players(q_player, random_player, num_games=100)
    #
    # print("\nEvaluating Q-Learning vs MCTS:")
    # mcts_player = MCTSPlayer('O')
    # wins, games = evaluate_players(q_player, mcts_player, num_games=100)
    #
    # # You can do more analysis with the 'games' data if needed
    # # For example, to find the longest and shortest games:
    # longest_game = max(games, key=lambda x: x[1])
    # shortest_game = min(games, key=lambda x: x[1])
    # print(f"\nLongest game: {longest_game[1]} moves, Winner: {longest_game[0]}")
    # print(f"Shortest game: {shortest_game[1]} moves, Winner: {shortest_game[0]}")

    """
    DQN Training
    """
    dqn_agent = DQNAgent(symbol='X', input_channels=3, action_size=81, memory_size=10000, batch_size=64, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99995, learning_rate=0.001)
    random_opponent = RandomPlayer('O')
    trained_agent = train_dqn_agent(dqn_agent, random_opponent, num_episodes=10000)
    trained_agent.save_model('dqn_u3_model.pth')
