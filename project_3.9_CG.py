import numpy as np
import matplotlib.pyplot as plt

def game_of_life(A: np.ndarray):
    
    rows, cols = np.array(A).shape
    new_state = np.zeros_like(A)

    # Loop through each cell in the grid
    for x in range(rows):
        for y in range(cols):
            # Count the live neighbors of the current cell
            live_neighbors = count_live_neighbors(A, x, y)

            # Apply the rules of the game
            if A[x, y] == 1:  # If the cell is alive
                if live_neighbors == 2 or live_neighbors == 3:
                    new_state[x, y] = 1  # Cell survives
            else:  # If the cell is dead
                if live_neighbors == 3:
                    new_state[x, y] = 1  # Cell comes to life

    return new_state

def count_live_neighbors(A: np.ndarray, x: int, y: int) -> int:
    rows, cols = np.array(A).shape
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the current cell
            nx = (x + dx) % rows  # Torus wrapping for rows
            ny = (y + dy) % cols  # Torus wrapping for columns
            count += A[nx, ny]
    return count

def animate(A: np.ndarray, steps: int):
    plt.figure()
    for _ in range(steps):
        plt.matshow(A)
        plt.show()
        A = game_of_life(A)

# Test the program with different starting states
if __name__ == "__main__":
    # Blinker pattern
    blinker = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
    
    # Glider pattern
    glider = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 1]])

    # Gospers glider gun
    from engcom.data import game_of_life_starts
    gosper_glider = game_of_life_starts("gosper_glider")

    # Call the game_of_life function for each starting state
    for name, state in [("Blinker", blinker), ("Glider", glider), 
                        ("Gosper Glider", gosper_glider)]:
        print("Starting state: {name}")
        plt.matshow(state)
        plt.show()

        # Animate the game for 10 steps
        animate(state, steps=10)


