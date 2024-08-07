import tkinter as tk
from tkinter import messagebox, ttk
from UT3 import U3
from Player import *

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultimate Tic-Tac-Toe")
        self.game = U3()  # Initializes the game logic
        self.setup_screens()

    def setup_screens(self):
        # Player selection screen setup
        self.selection_screen = tk.Frame(self.root)
        self.selection_screen.grid(row=0, column=0, sticky="nsew")
        self.setup_player_selection()

        # Game board screen setup (initially hidden)
        self.game_screen = tk.Frame(self.root)
        self.game_screen.grid(row=0, column=0, sticky="nsew")
        self.game_screen.grid_remove()

        self.buttons = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.create_widgets()

    def setup_player_selection(self):
        self.player1_type = tk.StringVar()
        self.player2_type = tk.StringVar()
        player_options = ["Human", "Random AI", "MCTS AI", "Q-Learning AI", "Minimax AI"]

        ttk.Label(self.selection_screen, text="Player 1:").grid(row=0, column=0)
        ttk.Combobox(self.selection_screen, textvariable=self.player1_type, values=player_options).grid(row=0, column=1)
        self.player1_type.set("Human")

        ttk.Label(self.selection_screen, text="Player 2:").grid(row=1, column=0)
        ttk.Combobox(self.selection_screen, textvariable=self.player2_type, values=player_options).grid(row=1, column=1)
        self.player2_type.set("MCTS AI")

        start_button = tk.Button(self.selection_screen, text="Start Game", command=self.start_game)
        start_button.grid(row=2, column=0, columnspan=2)

    def start_game(self):
        self.player1 = self.create_player(self.player1_type.get(), 'X')
        self.player2 = self.create_player(self.player2_type.get(), 'O')
        self.current_player = self.player1
        self.selection_screen.grid_remove()
        self.game_screen.grid()
        self.game = U3()
        self.update_board()
        self.check_ai_move()

    def create_player(self, player_type, symbol):
        if player_type == "Human":
            return HumanPlayer(symbol)
        elif player_type == "Random AI":
            return RandomPlayer(symbol)
        elif player_type == "MCTS AI":
            return MCTSPlayer(symbol)
        elif player_type == "Q-Learning AI":
            return QLearningPlayer(symbol)
        elif player_type == "Minimax AI":
            return MinimaxPlayer(symbol)
        else:
            raise ValueError("Unsupported player type selected")

    def create_widgets(self):
        for big_row in range(3):
            for big_col in range(3):
                frame = tk.Frame(self.game_screen, borderwidth=1, relief="solid")
                frame.grid(row=big_row * 3, column=big_col * 3, padx=0, pady=0)
                for small_row in range(3):
                    for small_col in range(3):
                        button = tk.Button(frame, text=' ', width=5, height=2,
                                           command=lambda br=big_row, bc=big_col, sr=small_row, sc=small_col: self.on_click(br, bc, sr, sc))
                        button.grid(row=small_row, column=small_col, padx=1, pady=1)
                        self.buttons[big_row][big_col][small_row][small_col] = button

    def on_click(self, big_row, big_col, small_row, small_col):
        if not self.game.game_over and isinstance(self.current_player, HumanPlayer) and self.game.make_move(big_row, big_col, small_row, small_col):
            self.switch_player()
            self.update_board()
            self.check_ai_move()

    def check_ai_move(self):
        # Check if the current player is AI and if the game is not over
        if isinstance(self.current_player,
                      (RandomPlayer, MCTSPlayer, QLearningPlayer, MinimaxPlayer)) and not self.game.game_over:
            self.make_ai_move()

    def make_ai_move(self):
        # Call the AI player's method to select a move
        move = self.current_player.make_move(self.game)
        if move:
            # Execute the move on the game logic
            self.game.make_move(*move)
            # Update the board to reflect the move
            self.update_board()
            # Switch to the next player
            self.switch_player()
            # Trigger the next move check in case of consecutive AI players
            self.root.after(250, self.check_ai_move)  # Adds a small delay for better visual effect

    def switch_player(self):
        # Switch between player1 and player2
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
        # After switching, immediately check if the new player is an AI
        self.check_ai_move()

    def update_board(self):
        for big_row in range(3):
            for big_col in range(3):
                sub_board_winner = self.game.meta_board[big_row][big_col]
                if sub_board_winner != self.game.place_holder:
                    symbol_to_display = sub_board_winner if sub_board_winner != 'D' else 'Draw'
                    for small_row in range(3):
                        for small_col in range(3):
                            self.buttons[big_row][big_col][small_row][small_col].config(text=symbol_to_display, state='disabled')
                else:
                    for small_row in range(3):
                        for small_col in range(3):
                            cell_value = self.game.board[big_row][big_col][small_row][small_col]
                            self.buttons[big_row][big_col][small_row][small_col].config(text=cell_value, state='normal')
        if self.game.game_over:
            winner_text = f"The winner is {self.game.winner}" if self.game.winner else "It's a draw!"
            messagebox.showinfo("Game Over", winner_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
