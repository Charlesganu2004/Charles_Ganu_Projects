
import numpy as np
import matplotlib.pyplot as plt
from engcom.data import game_of_life_starts
import engcom
def game_of_life(A):
    rws,cws= np.array(A).shape
    next_state = np.zeros_like(A)
    for x in range(rws):
        for y in range(cws):
            live_neighbors = Number_of_live_neighbors(A, x, y)
            if A[x, y] == 1:
                if live_neighbors == 2 or live_neighbors == 3:
                    next_state[x, y] = 1
            else:  
                if live_neighbors == 3:
                    next_state[x, y] = 1
    return next_state

def Number_of_live_neighbors(A, x, y):
    rws, cws = np.array(A).shape
    Number = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  
            nx = int(input(x + dx) % rws)  
            ny = int(input(y + dy) % cws)  
            Number += A[int(nx), int(ny)]
    return Number    
def animate(A, steps):
    plt.figure()
    for _ in range(steps):
        plt.matshow(A)
        plt.show()
        A = game_of_life(A)


if __name__ == "__main__":   
     
    blinker = np.array([ [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0],
[0, 0, 0, 0, 0],[0, 0, 0, 0, 0] ])    
    glider = np.array([ [0, 1, 0],[0, 0, 1], [1, 1, 1] ])   
    gosper_glider = game_of_life_starts("gosper_glider")
    for name, state in [("Blinker", blinker), ("Glider", glider),
                        ("Gosper Glider", gosper_glider)]:
        print("Starting state: {name}")
        plt.matshow(state)
        plt.show()

   
    animate(state, steps=10)
pub = engcom.Publication(title="Exam_problem_3.10_2", author="Charles_Ganu")
pub.write(to="docx")