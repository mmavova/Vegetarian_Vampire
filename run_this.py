#!/usr/bin/python3
import veg_vampire as vv
import numpy as np

# empty square = 0, Blood Root = 1, Crimson Pod = 2, Haemo Bush = 3, Red Shrub = 4, Sang Vine = 5.
board = np.array([[2, 3, 4, 1, 1],
                  [4, 5, 3, 2, 3],
                  [2, 4, 3, 3, 5],
                  [4, 1, 4, 3, 4],
                  [3, 1, 3, 5, 5]])

#recipes = np.array([[1, 1, 5, 1, 5], # BR - BR - SV - BR - SV
#                    [2, 2, 2, 3, 5], # CP - CP - CP - HB - SV
#                    [5, 4, 5, 4, 1], # SV - RS - SV - RS - BR
#                    [3, 1, 4, 2, 4], # HB - BR - RS - CP - RS
#                    [3, 2, 2, 3, 4]]) # HB - CP - CP - HB - RS



board = np.array([[1, 2, 2, 2, 2],
                  [1, 2, 2, 2, 2],
                  [1, 2, 2, 2, 2],
                  [5, 2, 2, 2, 2],
                  [5, 2, 2, 2, 2]])
attributes = {'have_ENDISWAL': True}
solutions = vv.solve_puzzle(vv.Puzzle(board, [0] * 5, attributes))
print(solutions)


for i in range(50):
    p = vv.Puzzle(np.random.randint(1, 6, size=[5, 5], dtype=int), [0] * 5, attributes)
    p.print_puzzle_state()
    solutions = vv.solve_puzzle(p)
    for s in solutions:
        print(s.recipes_achieved)
    
quit()

