# empty square = 0, Blood Root = 1, Crimson Pod = 2, Haemo Bush = 3,
# Red Shrub = 4, Sang Vine = 5.
# Recipes: Health Broth = 0, Magi Soup = 1, Strength Syrup = 2,
# Caramel Conversions = 3, Cake of Attack Bonus = 4
# The facts that the board is 5x5, recipes are length 5,
# and at most 5 recipes can be done are hardcoded all other the code.

# Limitations:
# * Half-dragon/consecrated strike plant moves are not supported
# * No glyph support for now (no ENDISWAL, PISORF, WEYTWUT, WONAFYT)

# Honestly, I only use Numpy for [i, j] syntax. I could torch it anytime.
import numpy as np
import copy
from operator import add
# Recipe information
# I have two global variables: g_recipes and g_recipes_totals.
# Another global constant g_board_move_weights
g_recipes = np.array([[1, 1, 5, 1, 5],   # BR - BR - SV - BR - SV
                      [2, 2, 2, 3, 5],   # CP - CP - CP - HB - SV
                      [5, 4, 5, 4, 1],   # SV - RS - SV - RS - BR
                      [3, 1, 4, 2, 4],   # HB - BR - RS - CP - RS
                      [3, 2, 2, 3, 4]])  # HB - CP - CP - HB - RS

# These are the weights of moves used in sort_the_moves
# higher value means a more desirable move
g_board_move_weights = [[0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 2, 0, 1],
                        [0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0]]


def precompute_recipes_totals():
    recipes_totals = np.zeros([5, 5], dtype=int)
    for r in range(5):
        for i in range(5):
            recipes_totals[r, g_recipes[r, i] - 1] += 1
    return recipes_totals


# The number of times each plan occures in each recipe
g_recipes_totals = precompute_recipes_totals()


class Puzzle:
    def __init__(self, board, buf=[0] * 5, attributes=None):
        self.board = board.copy()
        self.buf = buf.copy()
        self.board_totals = [0] * 5
        self.compute_board_totals()
        if attributes:
            self.attributes = attributes.copy()
        else:
            self.attributes = dict()
        self.populate_default_attributes()

    @classmethod
    def from_puzzle(cls, puzzle):
        return cls(puzzle.board, puzzle.buf, puzzle.attributes)

    def copy(self):
        p = Puzzle.from_puzzle(self)
        return p

    def populate_default_attributes(self):
        default_attributes = {'have_ENDISWAL': False}
        self.attributes = {**default_attributes, **self.attributes}

    def verify_initial_state_correctedness(self):
        # Some sanity checks.
        # I do not verify that you actually could reach the plants that are missing.
        # So you can pass a puzzle that has a plant missing in the middle,
        # even though it is unreachable.
        # Complete verification is a bit more involved,
        # since you could be a dragon standing in the middle, trying to box yourself in.
        if len(self.buf) != 5:
            raise Exception("Incorrect buffer length")
        try:
            first_zero = self.buf.index(0)
        except ValueError:
            raise Exception("Buffer cannot be full at the beginning of the puzzle.")
        for ind in range(first_zero, 5):
            if self.buf[ind]:
                raise Exception("Buffer has to be filled from left to right without any gaps")
        if self.board.shape != (5, 5):
            raise Exception("Board must be a 5x5 numpy array")

        total = 0
        total += sum([1 for i in range(5) for j in range(5) if self.board[i, j] > 0])
        total += sum([1 for elem in self.buf if elem > 0])
        if total % 5 != 0:
            msg = "The starting number of plants must be divisible by 5.\n" \
                  "Check both the board and the buffer."
            raise Exception(msg)
        return True

    def print_puzzle_state(self):
        # print buffer on top:
        print(" ", end='')
        for i in range(5):
            print(self.buf[i], end='')
        print(" ")

        print("-------")
        for i in range(5):
            print("|", end='')
            for j in range(5):
                print(self.board[i, j], end='')
            print("|")
        print("-------")

    def is_square_reachable(self, i, j):
        # returns true if board[i, j] is reachable
        # Assumes empty squares have been generated correctly.
        # Empty squares and the entrance ((1, 4), (2, 4), (3, 4)) are always reachable.
        reachable_with_ENDISWAL = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                                   (4, 1), (4, 2), (4, 3), (4, 4), (0, 4)}
        if self.board[i, j] == 0 or (j == 4 and (i == 1 or i == 2 or i == 3)):
            return True
        if self.attributes['have_ENDISWAL'] and (i, j) in reachable_with_ENDISWAL:
            return True

        s = set()  # the set of all neighbours
        for ind1 in range(max(0, i - 1), min(5, i + 2)):
            for ind2 in range(max(0, j - 1), min(5, j + 2)):
                if ind1 == i and ind2 == j:
                    continue
                s.add((ind1, ind2))
        for n in s:
            if self.board[n] == 0:
                return True

        return False

    def is_every_square_reachable(self):
        for i in range(0, 5):
            for j in range(0, 5):
                if not self.is_square_reachable(i, j):
                    return False
        return True

    def compute_board_totals(self):
        for i in range(5):
            for j in range(5):
                if self.board[i, j] > 0:
                    self.board_totals[self.board[i, j] - 1] += 1

    def get_available_moves(self):
        # For a 5x5 board, returns the set of available moves and true/false.
        # returns True if all squares are reachable.
        # The empty squares are considered to be reachable. Same for squares (2, 5), (3, 5), (4, 5)
        all_moves = []
        for i in range(5):
            for j in range(5):
                if self.is_square_reachable(i, j) and self.board[i, j]:
                    all_moves.append((i, j))

        return all_moves

    def make_move(self, move, solution_so_far=[], solutions_achieved=[]):
        # Several things happen here:
        # 1) we make the move
        # 2) we append it to the solution_so_far
        # 3) we update solutions_achieved if the buffer got cleared as a result of this move
        self.board_totals[self.board[move] - 1] -= 1
        ind_first_zero = next(ind for ind in range(5) if self.buf[ind] == 0)
        self.buf[ind_first_zero] = self.board[move]
        if self.buf[-1] > 0:
            self.process_buffer(solution_so_far, solutions_achieved)
        self.board[move] = 0
        if solution_so_far:
            solution_so_far.moves.append(move)

    def process_buffer(self, solution_so_far=[], solutions_achieved=[]):
        # buf has to be full.
        # If it is a recipe, then update the solution.
        # In any case, clear the buf at the end.
        # Just do a brute-force loop cause it is probably the fastest
        for r in range(5):
            match = True
            for i in range(5):
                if g_recipes[r, i] != self.buf[i]:
                    match = False
                    break
            if match:
                if solution_so_far:
                    solution_so_far.recipes_achieved[r] += 1
                    if solutions_achieved:
                        solutions_achieved.add_solution_to_set(solution_so_far)
                break

        for i in range(5):
            self.buf[i] = 0

    def buf_begins_recipe(self):
        # This function returns:
        # num_non_zero: the number of plants in the buffer already
        # the_list: [3, 4] -- list of recipes that it begins. 0, 1, or 2 members
        # (no 3 recipes start from the same plant)
        # Do not call it if the buffer is empty
        assert(self.buf[0] != 0)
        num_non_zero = next((ind for ind in range(5) if self.buf[ind] == 0), 5)

        the_list = []
        for i in range(5):
            match = True
            for j in range(num_non_zero):
                if self.buf[j] != g_recipes[i, j]:
                    match = False
                    break
            if match:
                the_list.append(i)
        return num_non_zero, the_list

    def theo_find_position_of_this_plant(self, plant):
        # Finds any position of a plant, assuming they are all accessible
        # I am sure there is a more pythonic way to write this
        for i in range(5):
            for j in range(5):
                if self.board[i, j] == plant:
                    return (i, j)
        print("In theo_find_position_of_this_plant, plant = ", plant)
        self.print_puzzle_state()
        raise Exception("Could not find the plant")

    def verify_solution(self, solution):
        # Verifies that solution moves achieve the promised recipe list.
        puzzle = Puzzle.from_puzzle(self)  # make a copy so that we can modify it
        solution_so_far = Solution()
        ss = Solution_Set()
        for m in solution.moves:
            if m not in puzzle.get_available_moves():
                print("Tried move ", m, " in position")
                puzzle.print_puzzle_state()
                raise Exception("Illegal move in a found solution")
            puzzle.make_move(m, solution_so_far, ss)

        if solution.recipes_achieved != solution_so_far.recipes_achieved:
            puzzle.print_puzzle_state()
            print(solution.recipes_achieved)
            print("but achieved: ", solution_so_far.recipes_achieved)
            raise Exception("Nope, the moves you gave me do not achieve the solution claimed")


# End of Puzzle class
g_recipe_names = ['Health Broth', 'Magi Soup', 'Strength Syrup',
                  'Caramel Conversion', 'Cake of Attack Bonus']
g_recipe_names_plural = ['Health Broths', 'Magi Soups', 'Strength Syrups',
                         'Caramel Conversions', 'Cake of Attack Bonuses']


def move_to_str(move):
    return "cut plant at (%s, %s)" % (move[0] + 1, move[1] + 1)


class Solution:
    def __init__(self, recipes_achieved=[], moves=[]):
        if recipes_achieved:
            self.recipes_achieved = recipes_achieved.copy()
        else:
            self.recipes_achieved = [0] * 5
        self.moves = moves.copy()

    def __repr__(self):
        return "<Solution: recipes_achieved:%s moves:%s>" % (self.recipes_achieved, self.moves)

    def __str__(self):
        res = ""
        if self.recipes_achieved:
            res += "Recipes achieved: "
            if sum(self.recipes_achieved):
                for i in range(5):
                    if self.recipes_achieved[i]:
                        res = res + str(self.recipes_achieved[i]) + " "
                        if self.recipes_achieved[i] > 1:
                            res += g_recipe_names_plural[i]
                        else:
                            res += g_recipe_names[i]
                        if sum([self.recipes_achieved[j] for j in range(i + 1, 5)]):
                            res += ", "
                        else:
                            res += ". "
                            break
            else:
                res += "Achieved Nothing. "
        res += "Moves:\n"
        if self.moves:
            for i in range(len(self.moves)):
                res += move_to_str(self.moves[i])
                if i < len(self.moves) - 1:
                    res += ", "
                else:
                    res += ". "

        else:
            res += "none. "
        return res

    def __lt__(self, other):
        # I am not considering length/complexity of moves yet
        return sum(self.recipes_achieved) < sum(other.recipes_achieved)

    def add_move(self, move, recipe_achieved=None):
        self.moves.append(move)
        if recipe_achieved is not None:
            self.recipes_achieved[recipe_achieved] += 1

    def copy(self):
        obj = Solution(self.recipes_achieved, self.moves)
        return obj

    def trim_solution(self, puzzle):
        # This function removes chaff at the end of the solution
        # that does not actually achieve any recipes.
        puzzle = puzzle.copy()
        ssf = Solution()  # solution_so_far
        for i in range(len(self.moves)):
            puzzle.make_move(self.moves[i], ssf)
            if self.recipes_achieved == ssf.recipes_achieved:
                break
        del self.moves[i + 1:]


# End of Solution class

class Solution_Set:
    def __init__(self, solutions=[]):
        self.solutions = copy.deepcopy(solutions)
        self.exclude_dominated_solutions()

    def __getitem__(self, key):
        return self.solutions[key]

    def __str__(self):
        if not self.solutions:
            return "Solution set is empty. "
        res = "%s different optimal solutions." % len(self.solutions)
        for s in self.solutions:
            res = res + "\n" + s.__str__()
        return res

    def exclude_dominated_solutions(self, solutions_achieved=[]):
        # Removes solutions that are not optimal
        # Also, if solutions_achieved is provided,
        # then we remove all solutions dominated by solutions_achieved
        for sol1 in self.solutions.copy():
            for sol2 in self.solutions.copy():
                if sol1 is sol2:
                    continue
                if theo_is_dominated_by(sol2.recipes_achieved, sol1.recipes_achieved):
                    if sol2 in self.solutions:  # possibly it was removed earlier
                        self.solutions.remove(sol2)
                        continue
        if solutions_achieved:
            for sol1 in solutions_achieved.solutions:
                for sol2 in self.solutions.copy():
                    if theo_is_dominated_by(sol2.recipes_achieved, sol1.recipes_achieved):
                        if sol2 in self.solutions:  # possibly it was removed earlier
                            self.solutions.remove(sol2)
                            continue

    def add_solution_to_set(self, new_solution):
        # we first check that
        # the new solution is not dominated by any solution in the set
        # Then we remove all the solutions that are dominated by the new one.
        for s in self.solutions:
            if theo_is_dominated_by(new_solution.recipes_achieved, s.recipes_achieved):
                return

        for s in self.solutions.copy():
            if theo_is_dominated_by(s.recipes_achieved, new_solution.recipes_achieved):
                self.solutions.remove(s)
        self.solutions.append(new_solution)

    def sort_by_total(self):
        self.solutions.sort(reverse=True)


# end of Solutions_Set class

# class Solver:
#         def __init__(self):
#                 pass

''' Solver proper code starts here.
Prefix theo_ means that this function assumes every square is reachable.
theo_solution means just an array of 5x1 that counts how many recipes of each kind can be done.

We start with theo_ functions that return theo_solutions only.
This is abstract stuff that only really cares about the board_totals and the recipes' definition.
'''


def theo_is_dominated_by(ts1, ts2):
    # Takes two arrays of equal size. Returns true if ts1(i) <= ts2(i) for each i.
    # return ((ts2 - ts1) >= 0).all() # this is numpy-style
    return all([pair[0] <= pair[1] for pair in zip(ts1, ts2)])


def theo_is_a_theo_solution(x, board_totals):
    # x is the number of recipes of each type
    # Returns true if x is a theoretical solution of recipes for board_totals.
    for i in range(5):  # for each plant type check that we have enough of it
        this_sum = 0
        for j in range(5):
            this_sum += x[j] * g_recipes_totals[j, i]
            # print('i = {0}: this_sum = {1}, board = {2}'.format(i, this_sum, board_totals))
        if this_sum > board_totals[i]:
            return False
    return True


def theo_list_optimal_theo_solutions_no_buf(board_totals):
    # Returns the list of all possible theo solutions, assuming that all plants are reachable.

    # First find the max for each recipe, then do brute-force loop.
    # For higher # of coordinates recursively computing
    # the max of the remaining ones is, ofc, optimal and faster.
    # I could optimize this function later if needed
    s = []
    theo_max_for_each_recipe = [0] * 5

    for r in range(5):
        this_solution = [0] * 5
        for i in range(5, 0, -1):
            this_solution[r] = i
            if theo_is_a_theo_solution(this_solution, board_totals):
                s.append(this_solution)
                theo_max_for_each_recipe[r] = i
                break

    # Now the brute-force loop
    # print('max: ', theo_max_for_each_recipe)
    for i1 in range(theo_max_for_each_recipe[0], -1, -1):
        for i2 in range(theo_max_for_each_recipe[1], -1, -1):
            for i3 in range(theo_max_for_each_recipe[2], -1, -1):
                for i4 in range(theo_max_for_each_recipe[3], -1, -1):
                    for i5 in range(theo_max_for_each_recipe[4], -1, -1):
                        this_solution = [i1, i2, i3, i4, i5]
                        # print('considering combo: ', this_solution)
                        if sum(this_solution) == 0:
                            continue
                        if theo_is_a_theo_solution(this_solution, board_totals):
                            is_dominated = False
                            for sol in s:
                                if theo_is_dominated_by(this_solution, sol):
                                    is_dominated = True
                                    break
                            if is_dominated:
                                break
                            # print('adding solution ', this_solution)
                            s.append(this_solution)
                            break

    # print(s)
    # We can still have non-optimal solutions on the list at this point:
    # s = exclude_dominated_solutions(s)
    for sol1 in s.copy():
        for sol2 in s.copy():
            if sol1 is sol2:
                continue
            if theo_is_dominated_by(sol2, sol1):
                s.remove(sol2)
                break
    return s


def theo_list_optimal_theo_solutions(puzzle):
    # In this function we use the fact that the board size = buf size * 5,
    # and every recipe is length 5.
    # Otherwise the logic in this function does not hold.
    # If buf is empty: use theo_list_optimal_theo_solutions_no_buf.
    # else:
    #         1) check if buf begins any recipe,
    #         2) check that we have enough plants to complete the recipe
    #         3) after that call theo_list_optimal_theo_solutions_no_buf
    #            on the remaining total and add 1 recipe.
    #         4) join the solutions from step 3 and from
    #            theo_list_optimal_theo_solutions_no_buf(board_totals)

    solutions = theo_list_optimal_theo_solutions_no_buf(puzzle.board_totals)
    if puzzle.buf[0]:
        # num_non_zero: the number of plants in the buffer already
        # the_list: [3, 4] -- list of recipes that it begins.
        num_non_zero, the_list = puzzle.buf_begins_recipe()
        for r_ind in the_list:  # buf begins this recipe
            # check that there are enough plants to build it
            new_totals = puzzle.board_totals.copy()
            for j in range(num_non_zero, 5):
                p = g_recipes[r_ind, j]
                new_totals[p - 1] -= 1
            if min(new_totals) < 0:  # this recipe cannot be completed
                continue
            # this recipe _can_ has been completed
            these_sols = theo_list_optimal_theo_solutions_no_buf(new_totals)
            if not these_sols:
                these_sols = [[0] * 5]
            # need to add r_ind to it
            for s in these_sols:
                s[r_ind] += 1
            solutions.extend(these_sols)
    return solutions


def theo_update_theo_solutions_remaining(theo_solutions_remaining, solutions_achieved):
    # Removes theo_solutions that are dominated by real solutions already achieved
    for s in solutions_achieved:
        for t in theo_solutions_remaining.copy():
            if theo_is_dominated_by(t, s.recipes_achieved):
                theo_solutions_remaining.remove(t)


def theo_complete_this_recipe_solution(puzzle, solution_so_far, recipe_num):
    # if puzzle.buf is not empty, then it has to be the beginning of recipe_num.
    # adds the moves that produce recipe_num to solution_so_far
    # puzzle, solutions_so_far are all modified!
    for i in range(5):  # loop over the recipe
        if puzzle.buf[i]:
            if i == 4:
                raise Exception("Huh? This should never happen")
            if puzzle.buf[i] != g_recipes[recipe_num, i]:
                print("buf: ", puzzle.buf, ", recipe num = ", recipe_num)
                print("recipes: ", g_recipes)
                raise Exception("Again, check your code, dunderhead")
            continue
        # else:
        plant = g_recipes[recipe_num, i]
        m = puzzle.theo_find_position_of_this_plant(plant)
        puzzle.make_move(m, solution_so_far)


'''
If current buffer cannot be completed to a full recipe, we fill it up with useless plants
But we need to be careful to make sure these plants are useless
recipe_list is a 5x1 array of the recipe totals we plan to achieve with the remainder of the board.
'''


def theo_find_filler_moves(puzzle, solution_so_far, recipe_list):
    # First compute the unused plants assuming we completed all recipes
    # Then we can pick something out of those to fill the buffer.
    remainder = puzzle.board_totals.copy()
    for i in range(5):  # loop over all recipes
        for j in range(5):
            # can I vectorize this when I remove numpy?
            remainder[j] -= recipe_list[i] * g_recipes_totals[i, j]
    # print("remainder: ", remainder)
    # sanity check, can disable later:
    for i in range(5):
        if remainder[i] < 0:
            raise Exception("Damn, another bug")

    # Ready to fill, baby
    for i in range(5):
        if puzzle.buf[i]:
            continue
        plant_ind = remainder.index(max(remainder))
        move = puzzle.theo_find_position_of_this_plant(plant_ind + 1)
        puzzle.make_move(move, solution_so_far)
        remainder[plant_ind] -= 1


def theo_find_moves_for_this_theo_solution(puzzle, recipe_list):
    # Here recipe_list is a 5x1 array
    # The function returns a Solution attaining the recipe_list
    # the solution must be reachable
    recipe_list = recipe_list.copy()
    solution_so_far = Solution()

    # first, deal with the buffer
    if puzzle.buf[0]:
        num_non_zero, the_list = puzzle.buf_begins_recipe()
        # we don't care which recipe it begins, just get one.
        just_one = [r_ind for r_ind in the_list if recipe_list[r_ind] > 0]
        if just_one:
            theo_complete_this_recipe_solution(puzzle, solution_so_far, just_one[0])
            recipe_list[just_one[0]] -= 1
        else:
            theo_find_filler_moves(puzzle, solution_so_far, recipe_list)
            # print("After filler moves:")
            # print_puzzle_state(board, buf)
    # Now buf is empty
    # If recipe_list is [1, 0, 2, 0, 1]
    # recipe_list_literal is gonna be [0, 2, 2, 4]
    recipe_list_literal = []
    for j in range(5):
        if recipe_list[j]:
            recipe_list_literal.extend([j] * recipe_list[j])
    # print("remaining recipes: ", recipe_list_literal, "buf: ", buf)
    for r in recipe_list_literal:
        theo_complete_this_recipe_solution(puzzle, solution_so_far, r)
    return solution_so_far.moves


def theo_solve_puzzle(puzzle, solution_so_far, solutions_achieved):
    # This returns a Solution_Set
    theo_solutions = theo_list_optimal_theo_solutions(puzzle)
    # print("theo_solutions:", theo_solutions)
    if not theo_solutions:  # cannot extend the current solution
        if sum(solution_so_far.recipes_achieved):  # solution_so_far actually achieved something:
            return Solution_Set([solution_so_far])
        else:
            return Solution_Set()

    solutions = Solution_Set()
    for s in theo_solutions:
        # print("looping over theo_solutions, s = ", s)
        new_sol = solution_so_far.copy()
        # If you add a constant to every set in a set of optimal solutions,
        # the solutions stay optimal
        new_sol.recipes_achieved = list(map(add, new_sol.recipes_achieved, s))
        # print("sol so far + s = ", new_sol.recipes_achieved)
        for old_s in solutions_achieved:
            if theo_is_dominated_by(new_sol.recipes_achieved, old_s.recipes_achieved):
                # print("this one is dominated by ", old_s.recipes_achieved)
                continue
        moves = theo_find_moves_for_this_theo_solution(puzzle.copy(), s)
        new_sol.moves.extend(moves)
        solutions.add_solution_to_set(new_sol)
    return solutions


def solve_puzzle(puzzle):
    # buf (buffer) must be a 5x1 array of ints
    # The function will return a set of reachable solutions.
    # Dominated solutions (strictly worse than some other solution) are not returned.
    # This is the main function that users should call.
    solutions = Solution_Set()
    if not isinstance(puzzle, Puzzle):
        print("puzzle should be an instance of the Puzzle class")
        return solutions
    if not puzzle.verify_initial_state_correctedness():
        print("Incorrect puzzle")
        return solutions

    solutions = solver_solve_puzzle(puzzle.copy())
    # print("final solutions: ", solutions.solutions)
    for s in solutions.solutions:
        puzzle.verify_solution(s)
        s.trim_solution(puzzle)

    solutions.sort_by_total()
    return solutions


def solver_solve_puzzle(puzzle):
    theo_solutions_remaining = theo_list_optimal_theo_solutions(puzzle)
    solutions = Solution_Set()
    if not puzzle.attributes['have_ENDISWAL']:
        solve_puzzle_iteratively(puzzle, Solution(), solutions, theo_solutions_remaining)
    else:
        puzzle1 = puzzle.copy()
        puzzle1.attributes['have_ENDISWAL'] = False
        # Solve without ENDISWAL first
        solve_puzzle_iteratively(puzzle1, Solution(), solutions, theo_solutions_remaining)
        rem = len(theo_solutions_remaining)
        solve_puzzle_iteratively(puzzle,  Solution(), solutions, theo_solutions_remaining)
        if rem > len(theo_solutions_remaining):
            print("Found ", rem - len(theo_solutions_remaining), " solutions with ENDISWAL")
    return solutions


def solve_puzzle_iteratively(puzzle, solution_so_far, solutions_achieved, theo_solutions_remaining):
    # First version. No hashing. Just a few heuristics
    # print("")
    # print("Entering solve_puzzle_iteratively")
    # print("solution so far:", solution_so_far)
    # print("solutions:",  solutions_achieved)
    # print("remaining to be found: ", theo_solutions_remaining)
    if not theo_solutions_remaining:
        return
    remaining_theo_possible = theo_list_optimal_theo_solutions(puzzle.copy())
    if not remaining_theo_possible:  # no solutions
        # print("No theo solutions left, terminating this branch")
        return
    theo_solutions_plus = [list(map(add, ts, solution_so_far.recipes_achieved))
                           for ts in remaining_theo_possible]
    theo_update_theo_solutions_remaining(theo_solutions_plus, solutions_achieved)
    if not theo_solutions_plus:
        return

    if puzzle.is_every_square_reachable():
        # print("Every square is reachable")
        more_solutions = theo_solve_puzzle(puzzle.copy(), solution_so_far.copy(),
                                           solutions_achieved)
        for new_sol in more_solutions:
            solutions_achieved.add_solution_to_set(new_sol)
        # print("more_solutions: ", more_solutions)
        theo_update_theo_solutions_remaining(theo_solutions_remaining, solutions_achieved)
        return

    moves = puzzle.get_available_moves()
    # print("unsorted moves:", moves)
    # moves that complete/extend a recipe or increase
    # the number of reachable squares go to the front of the line
    moves = sort_the_moves(puzzle, moves)
    # print("sorted moves:", moves)
    for move in moves:
        m_puzzle = puzzle.copy()
        m_solution_so_far = solution_so_far.copy()
        # print("making move ", move)
        m_puzzle.make_move(move, m_solution_so_far, solutions_achieved)
        # print("solution so far:", m_solution_so_far)
        solve_puzzle_iteratively(m_puzzle, m_solution_so_far, solutions_achieved,
                                 theo_solutions_remaining)
        theo_update_theo_solutions_remaining(theo_solutions_remaining, solutions_achieved)
        if not theo_solutions_remaining:
            return


def sort_the_moves(puzzle, moves):
    # To the front of the line:
    # 1) moves that extend a recipe
    # 2) moves that start a recipe
    # 3) moves that unlock more moves.

    # the third coordinate is gonna be the weight of the move.
    move_weights = [0] * len(moves)
    # new_moves = moves.copy()
    # new_moves = [[*m, 0] for m in moves]

    # check for extending a recipe first
    if puzzle.buf[0] != 0:
        num_non_zero, the_list = puzzle.buf_begins_recipe()
        if the_list:
            for r in the_list:
                for m in range(len(moves)):
                    if g_recipes[r, num_non_zero] == puzzle.board[moves[m]]:
                        # print("adding ", num_non_zero, " to move ", m)
                        move_weights[m] += num_non_zero
    else:
        # Now check for moves that start a recipe
        for r in range(5):
            for m in range(len(moves)):
                if g_recipes[r, 0] == puzzle.board[moves[m]]:
                    move_weights[m] += 2  # Magic constant.

    # Ugh, I am too lazy to find unlocking moves for now.
    # Just adding extra weight to promising moves
    for m in range(len(moves)):
        move_weights[m] += g_board_move_weights[moves[m][0]][moves[m][1]]
    # print("sorting moves ", moves)
    # print("weights: ", move_weights)
    new_moves = [x for _, x in sorted(zip(move_weights, moves),
                                      key=lambda pair: pair[0], reverse=True)]
    return new_moves

# End of Solver class
