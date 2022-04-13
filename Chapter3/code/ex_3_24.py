from enum import Enum
import numpy as np
import sympy as sp
import pandas as pd

UP=1
DOWN=2
LEFT=3
RIGHT=4

GAMMA = 0.9

pistar = [
    [RIGHT, DOWN, LEFT, UP, LEFT], 
    [UP, UP, UP, LEFT, LEFT], 
    [UP for _ in range(5)],
    [UP for _ in range(5)],
    [UP for _ in range(5)]
]

def action(x, y, dir):
    if x == 0 and y == 1:
        return 4, 1, 10

    if x == 0 and y == 3:
        return 2, 3, 5

    if dir == UP:
        x -= 1
    elif dir == DOWN:
        x += 1
    elif dir == LEFT:
        y -= 1
    elif dir == RIGHT:
        y += 1
    return x, y, 0

def get_valuestar(x, y):
    cur_gamma = 1
    cur_value = 0
    for i in range(1000):
        x, y, r = action(x, y, pistar[x][y])
        cur_value += cur_gamma * r
        cur_gamma *= GAMMA
    return cur_value
        
values = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        values[i][j] = get_valuestar(i, j)

print(pd.DataFrame(values).to_latex(float_format="{:.3f}".format))
print(values)

