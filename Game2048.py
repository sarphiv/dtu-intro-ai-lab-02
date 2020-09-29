# 2048 game

# Import libraries used for this program
# SOURCE: Mikkel Nørgaard Schmidt
# DATE: 2020-09-30
# MODIFIED to remove "pygame" dependency

 
import numpy as np

#%%

class Game2048():    
    # Rendering?
    rendering = False

    
    def __init__(self, state=None):
        if state is None:
            self.board, self.score = self.new_game()
        else:
            board, score = state
            self.board, self.score = board.copy(), score

            
    def step(self, action):
        self.board, self.score = self.move(self.board, self.score, action)

        # return observation, reward, done
        done = self.game_over(self.board)        
        return ((self.board, self.score), self.score, done)
        
    def render(self):
        if not self.rendering:
            self.init_render()

    def reset(self):
        self.board, self.score = self.new_game()

    def close(self):
        pass
 
    def init_render(self):
        self.rendering = True


    def random_empty_pos(self, board):
        i, j = np.where(board==0)    
        k = np.random.randint(len(i))
        return (i[k], j[k])
    
    def game_over(self, board):
        # Space left on board?
        if not np.all(board):
            return False
        # Any neighbors that can be merged?
        for i in range(4):
            for j in range(3):
                if board[i,j]==board[i,j+1] or board[j,i]==board[j+1,i]:
                    return False
        return True
    
    def compress_left(self, board):
        for i in range(4):
            k = 0
            for j in range(4):
                if board[i,j]:
                    board[i,k] = board[i,j]
                    k += 1
            for j in range(k, 4):
                board[i,j] = 0       
        return board
    
    def move_left(self, board, score):
        # Remove zeros        
        board = self.compress_left(board)
        
        # Combine adjacent values (replace second val by zero)
        for i in range(4):
            for j in range(3):
                if board[i,j]==board[i,j+1] and board[i,j] != 0:
                    board[i,j] *= 2                    
                    board[i,j+1] = 0
                    score += board[i,j]
        # Remove zeros                    
        board = self.compress_left(board)    
        
        return (board, score)
    
    def move(self, board, score, direction='left'):
        # Save initial board
        initial_board = board.copy()
    
        # Rotate board, move left, rotate back
        if direction=='left':
            board, score = self.move_left(board, score)
        elif direction=='right':
            board = board[:,::-1]
            board, score = self.move_left(board, score)
            board = board[:,::-1]
        elif direction=='up':
            board = board.T
            board, score = self.move_left(board, score)
            board = board.T
        elif direction=='down':
            board = board.T[:,::-1]
            board, score = self.move_left(board, score)
            board = board[:,::-1].T
    
        # Add new 2 or 4 at random pos (if board has changed and there is space)
        if not np.all(board) and np.any(board != initial_board):
            board[self.random_empty_pos(board)] = [2,4][np.random.randint(2)]       
            
        return (board, score)
       
    def new_game(self):
        board = np.zeros((4, 4), dtype=int)
        for k in range(2):
            i, j = self.random_empty_pos(board)
            board[i,j] = 2
        score = 0
        return (board, score)


