#!/usr/bin/python3
import veg_vampire as vv
import numpy as np

# empty square = 0, Blood Root = 1, Crimson Pod = 2, Haemo Bush = 3, Red Shrub = 4, Sang Vine = 5.
# difficult example to analyze later:
# big set of solutions
# buf = [0] * 5
# board = np.array([[2, 3, 4, 1, 1],
#                   [4, 5, 3, 2, 3],
#                   [2, 4, 3, 3, 5],
#                   [4, 1, 4, 3, 4],
#                   [3, 1, 3, 5, 5]])

#recipes = np.array([[1, 1, 5, 1, 5], # BR - BR - SV - BR - SV
#                    [2, 2, 2, 3, 5], # CP - CP - CP - HB - SV
#                    [5, 4, 5, 4, 1], # SV - RS - SV - RS - BR
#                    [3, 1, 4, 2, 4], # HB - BR - RS - CP - RS
#                    [3, 2, 2, 3, 4]]) # HB - CP - CP - HB - RS



board = np.array([[1, 1, 5, 0, 0],
                  [5, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
solutions = vv.solve_puzzle(vv.Puzzle(board, [0] * 5))
print(solutions)

for i in range(50):
    p = vv.Puzzle(np.random.randint(1, 6, size=[5, 5], dtype=int))
    p.print_puzzle_state()
    solutions = vv.solve_puzzle(p)
    for s in solutions:
        print(s.recipes_achieved)
    
quit()

board = np.array([[5, 1, 4, 4, 5],
                  [5, 1, 3, 1, 1],
                  [2, 1, 5, 5, 1],
                  [1, 2, 1, 2, 1],
                  [3, 1, 1, 1, 5]])


buf = [0, 0, 0, 0, 0]
# board = np.array([[3, 5, 2, 1, 1],
#                   [2, 4, 5, 3, 5],
#                   [3, 2, 5, 5, 4],
#                   [2, 1, 2, 5, 2],
#                   [5, 1, 5, 4, 4]])


solutions = vv.solve_puzzle(board, buf)
print(solutions)
quit()




#quit()



# board_totals = vv.compute_board_totals(board)
# solution_so_far = vv.empty_solution()
# move = (3, 4)
# vv.make_move(board, board_totals, buf, solution_so_far, move)
# vv.print_game_state(board, buf)
# moves = vv.get_available_moves(board)
# print(moves)



# buf = [3, 4, 0, 0, 0]
# board = np.array([[2, 0, 0, 0, 1],
#                   [0, 0, 0, 2, 0],
#                   [2, 0, 0, 0, 5],
#                   [0, 1, 4, 0, 4],
#                   [3, 1, 3, 5, 5]])


# solutions = vv.theo_solve_puzzle(board, vv.compute_board_totals(board), buf, vv.empty_solution(), [])
# print(solutions)
# quit()

# board = np.array([[2, 3, 4, 1, 1],
#                   [4, 0, 3, 0, 0],
#                   [2, 4, 0, 3, 0],
#                   [4, 0, 4, 0, 4],
#                   [3, 1, 3, 5, 5]])


# buf = [3, 2, 0, 0, 0]
# board = np.array([[0, 3, 4, 1, 1],
#                   [4, 0, 3, 0, 0],
#                   [2, 4, 0, 3, 0],
#                   [4, 1, 4, 0, 4],
#                   [3, 1, 3, 5, 5]])

#solutions = vv.theo_list_all_optimal_solutions(vv.compute_board_totals(board), buf, [])


#solutions = vv.theo_solve_puzzle(board, vv.compute_board_totals(board), buf, vv.empty_solution(), [])

solutions = vv.solve_puzzle(board, buf)
print(solutions)
quit()
#solutions = vv.theo_list_all_optimal_solutions([3, 0, 0, 0, 2], [0] * 5, [])


board = np.array([[1, 1, 5, 0, 0],
                  [5, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])

solution_so_far =  {'recipes': [0, 0, 0, 0, 0], 'moves': [(0, 1)]}
#solutions = vv.theo_solve_puzzle(board, [2, 0, 0, 0, 2], [1, 0, 0, 0, 0], solution_so_far, [])
#print(solutions)





# board_totals = [0, 0, 0, 0, 0]
# buf = [0, 0, 0, 0, 0]
solution_so_far = vv.empty_solution()
# move = (3, 4)


# vv.make_move(board, board_totals, buf, solution_so_far, move)

