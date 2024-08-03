class U3:
    def __init__(self, board=None, last_move=None, current_player=None):
        # Initialize the game board: 3x3 grid of 3x3 boards
        self.board = [[[[' ' for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)] if board is None else board
        self.current_player = 'X' if current_player is None else current_player # X always starts
        self.last_move = None if last_move is None else last_move  # Stores the last move made
        self.game_over = False  # Flag to check if the game has ended
        self.winner = None  # Stores the winner of the game
        self.moves = []
        self.free_move_from_last_move = False  # Flag for if the last move leads to a full/won small_board - choose your move everywhere
        self.place_holder = '~'  # a place_holder for an empy cell in a won small_board
        self.meta_board = [[self.place_holder for _ in range(3)] for _ in range(3)]

    def make_move(self, big_row, big_col, small_row, small_col):
        # Check if the game is already over
        if self.game_over:
            print('Game Over')
            return False

        # if it's a free move, check that the spot is not in a full or already won small_board
        if self.free_move_from_last_move:
            if self.check_small_board_win(big_row, big_col) or self.is_small_board_full(big_row, big_col):
                print('Invalid move in big row and big col, small_board is full or won')
                return False
            pass
        elif self.last_move and (big_row, big_col) != self.last_move: # Check if the move is in the correct big board based on the last move
            if not self.is_small_board_full(self.last_move[0], self.last_move[1]):
                print('Invalid move in big row and big col')
                return False

        # Check if the chosen cell is empty
        if self.board[big_row][big_col][small_row][small_col] != ' ':
            print('Invalid move in small row and small col')
            return False

        # Make the move
        self.board[big_row][big_col][small_row][small_col] = self.current_player
        self.moves.append((big_row, big_col, small_row, small_col))
        self.last_move = self.moves[-1][-2:]

        # Check if move leads to a full or won small_board - next player gets a free-choice move in the board
        if self.check_small_board_win(self.last_move[0], self.last_move[1]) or self.is_small_board_full(self.last_move[0], self.last_move[1]):
            self.free_move_from_last_move = True
        else:
            self.free_move_from_last_move = False

        # Check if the small board is won by the move. If won, fill empty cells with self.place_holder, update the meta board of the result, Check for a win in the small board and potentially the big board
        if self.check_small_board_win(big_row, big_col):
            # if the game in the small board is won, update the meta board of the result
            self.meta_board[big_row][big_col] = self.current_player
            # fill empty cells with self.place_holder
            for i in range(3):
                for j in range(3):
                    if self.board[big_row][big_col][i][j] == ' ':
                        self.board[big_row][big_col][i][j] = self.place_holder
            # Check for a win
            if self.check_big_board_win():
                self.game_over = True
                self.winner = self.current_player

        # Check if the board is full (draw)
        if self.is_board_full():
            self.game_over = True

        # Switch to the other player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def check_small_board_win(self, big_row, big_col):
        # Get the small board we're checking
        board = self.board[big_row][big_col]
        optional_values = [' ', self.place_holder]  # values representing empty cells ' ' or place_holder for empy cell in a won small_board

        won = False
        # Check rows and columns
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] and board[i][0] not in optional_values:
                won = True  # Row win
            if board[0][i] == board[1][i] == board[2][i] and board[0][i] not in optional_values:
                won = True  # Column win

        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] not in optional_values:
            won = True  # Main diagonal win
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] not in optional_values:
            won = True  # Other diagonal win

        return won

    def check_big_board_win(self):
        board = self.meta_board
        won = False
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] and row[0] not in [self.place_holder, ' ']:
                won = True

        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] and board[0][col] not in [self.place_holder, ' ']:
                won = True

        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] and board[0][0] not in [self.place_holder, ' ']:
            won = True
        if board[0][2] == board[1][1] == board[2][0] and board[0][2] not in [self.place_holder, ' ']:
            won = True

        # No win found
        return won


    def is_small_board_full(self, big_row, big_col):
        # Check if all cells in a small board are filled (including self.place_holder)
        return all(self.board[big_row][big_col][i][j] != ' ' for i in range(3) for j in range(3))

    def is_board_full(self):
        # Check if all small boards are full
        return all(self.is_small_board_full(i, j) for i in range(3) for j in range(3))

    def get_valid_moves(self):
        # If it's the first move or the relevant small board is full or already won, any empty cell is valid -
        # empty cells in a won small_board are replaced with self.place_holder so the small board is considered full
        if not self.last_move or self.is_small_board_full(self.last_move[0], self.last_move[1]) or self.check_small_board_win(self.last_move[0], self.last_move[1]):
            return [(br, bc, sr, sc) for br in range(3) for bc in range(3)
                    for sr in range(3) for sc in range(3)
                    if self.board[br][bc][sr][sc] == ' ']
        else:
            # Otherwise, only empty cells in the relevant small board are valid
            br, bc = self.last_move
            return [(br, bc, sr, sc) for sr in range(3) for sc in range(3)
                    if self.board[br][bc][sr][sc] == ' ']


    def print_board(self, place_holder=True):
        def get_board_row_string(big_row, small_row, place_holder):
            row_string = ""
            for big_col in range(3):
                for small_col in range(3):
                    cell = self.board[big_row][big_col][small_row][small_col]
                    row_string += f"{cell if cell != ' ' else '·'}" if place_holder else f"{cell if cell not in [' ', self.place_holder] else '·'}"  # print with or without the place_holder
                row_string += "|"
            return row_string

        delimiter = "+---+---+---+\n"
        print(delimiter, end="")
        for big_row in range(3):
            for small_row in range(3):
                print(f"|{get_board_row_string(big_row, small_row, place_holder)}")
            if big_row < 2:
                print("+---+---+---+")
        print(delimiter, end="")

    def print_meta_board(self):
        print("+---+---+---+")
        for i, row in enumerate(self.meta_board):
            print("|", end="")
            for cell in row:
                print(f" {cell} |", end="")
            print()
            if i < 2:
                print("+---+---+---+")
        print("+---+---+---+")

if __name__ == '__main__':
    u3 = U3()
    moves = [
        (1, 1, 1, 1),
        (1, 1, 0, 1),
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 1, 0, 0),
        (0, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (1, 1, 0, 2),
        (0, 2, 1, 1),
        (1, 1, 1, 2),
        (1, 2, 0, 1),
        (0, 1, 2, 1),
        (2, 1, 1, 2),
        (1, 2, 1, 1),
        (1, 1, 2, 2),
        (2, 2, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 2),
        (0, 2, 0, 1),
        (0, 1, 1, 2),
        (1, 2, 0, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 1),
        (2, 1, 2, 2),
        (2, 2, 1, 1),
        (1, 2, 0, 2),
        (1, 2, 2, 0),
        (2, 0, 2, 2),
        (2, 2, 1, 2),
        (1, 2, 1, 0),
        (1, 0, 2, 0),
        (2, 0, 1, 2),
        (1, 2, 1, 2),
        (1, 2, 2, 1),
        (2, 1, 1, 0),
        (1, 0, 1, 2),
        (1, 2, 2, 2),
        (2, 2, 2, 0),
        (2, 0, 1, 1),
        (2, 0, 0, 0),
        (2, 0, 0, 2),
        (2, 1, 2, 0),
        (2, 0, 2, 0)
    ]

    print('STARTING THE GAME!!!')
    u3.print_board()
    for step, move in enumerate(moves):
        print('STEP NUMBER:', str(step+1))
        u3.make_move(*move)
        u3.print_board(place_holder=True)
    print('THE WINNER IS:', u3.winner)