import unittest
import numpy as np
import veg_vampire as vv


class Veg_Vampire_Test(unittest.TestCase):
    def test_theo_is_dominated_by(self):
        a1 = np.array([1, 2])
        a2 = np.array([3, 4])
        self.assertTrue(vv.theo_is_dominated_by(a1, a2))
        self.assertFalse(vv.theo_is_dominated_by(a2, a1))
        self.assertTrue(vv.theo_is_dominated_by(a1, a1))

    def test_theo_is_a_theo_solution(self):
        board_totals1 = np.array([4, 0, 0, 4, 4])
        self.assertTrue(vv.theo_is_a_theo_solution(np.array([1, 0, 0, 0, 0]), board_totals1))
        self.assertTrue(vv.theo_is_a_theo_solution(np.array([0, 0, 1, 0, 0]), board_totals1))
        self.assertTrue(vv.theo_is_a_theo_solution(np.array([1, 0, 1, 0, 0]), board_totals1))
        self.assertFalse(vv.theo_is_a_theo_solution(np.array([2, 0, 1, 0, 0]), board_totals1))
        board_totals2 = np.array([10, 9, 6, 10, 10])
        self.assertTrue(vv.theo_is_a_theo_solution(np.array([1, 1, 1, 1, 2]), board_totals2))
        self.assertFalse(vv.theo_is_a_theo_solution(np.array([1, 1, 4, 1, 2]), board_totals2))

    def test_theo_list_optimal_theo_solutions_no_buf(self):
        b = np.array([0, 0, 0, 0, 0])
        self.assertTrue(len(vv.theo_list_optimal_theo_solutions_no_buf(b)) == 0)
        b = np.array([1, 0, 0, 2, 2])
        self.assertTrue(vv.theo_list_optimal_theo_solutions_no_buf(b) == [[0, 0, 1, 0, 0]])

    def test_theo_list_optimal_theo_solutions(self):
        self.assertTrue(len(vv.theo_list_optimal_theo_solutions(
            vv.Puzzle(np.zeros([5, 5], dtype=int)))) == 0)
        p = vv.Puzzle(np.array([[1, 1, 1, 5, 5],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), [0] * 5)

        self.assertTrue(len(vv.theo_list_optimal_theo_solutions(p)) == 1)

    def test_theo_solve_puzzle(self):
        p = vv.Puzzle(np.array([[1, 0, 5, 0, 0],
                                [5, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]), [1, 0, 0, 0, 0])
        solutions = vv.theo_solve_puzzle(p, vv.Solution(), vv.Solution_Set())
        self.assertTrue(solutions[0].recipes_achieved == [1, 0, 0, 0, 0])

    def test_solve_puzzle(self):
        board = np.array([[1, 1, 5, 0, 0],
                          [5, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        solutions = vv.solve_puzzle(vv.Puzzle(board))
        self.assertTrue(solutions[0].recipes_achieved == [1, 0, 0, 0, 0])

        board = np.array([[5, 1, 4, 4, 5],
                          [5, 1, 3, 1, 1],
                          [2, 1, 5, 5, 1],
                          [1, 2, 1, 2, 1],
                          [3, 1, 1, 1, 5]])
        solutions = vv.solve_puzzle(vv.Puzzle(board))
        # got this from an early implementation, never validated
        solutions2 = [{'recipes': [3, 0, 0, 0, 1], 'moves': [(2, 4), (1, 3), (2, 2), (1, 1), (0, 0), (1, 2), (3, 1), (3, 3), (4, 0), (0, 2), (0, 1), (1, 4), (0, 4), (2, 1), (1, 0), (3, 0), (3, 2), (2, 3), (3, 4), (4, 4)]}, {'recipes': [2, 0, 1, 0, 0], 'moves': [(2, 4), (1, 3), (2, 2), (1, 1), (0, 0), (1, 2), (3, 1), (3, 3), (0, 1), (1, 4), (2, 1), (3, 0), (0, 4), (3, 2), (1, 0), (2, 3), (0, 2), (4, 4), (0, 3), (3, 4)]}, {'recipes': [3, 0, 0, 1, 0], 'moves': [(2, 4), (1, 3), (2, 2), (1, 1), (0, 0), (1, 2), (3, 1), (3, 3), (0, 1), (1, 4), (2, 1), (3, 0), (0, 4), (3, 2), (1, 0), (3, 4), (4, 1), (2, 3), (4, 2), (4, 4), (4, 0), (4, 3), (0, 2), (2, 0), (0, 3)]}, {'recipes': [1, 1, 1, 0, 0], 'moves': [(2, 4), (1, 3), (2, 2), (1, 1), (0, 0), (1, 2), (0, 1), (0, 4), (1, 4), (2, 1), (3, 0), (3, 2), (3, 4), (4, 1), (4, 2), (2, 0), (3, 1), (3, 3), (4, 0), (1, 0), (2, 3), (0, 2), (4, 4), (0, 3), (4, 3)]}, {'recipes': [2, 1, 0, 0, 0], 'moves': [(2, 4), (1, 3), (2, 2), (1, 1), (0, 2), (1, 2), (3, 2), (0, 3), (3, 4), (4, 1), (0, 1), (1, 4), (0, 0), (2, 1), (0, 4), (3, 0), (4, 2), (1, 0), (4, 3), (2, 3), (2, 0), (3, 1), (3, 3), (4, 0), (4, 4)]}]
        for s in solutions:
            found = False
            for s2 in solutions2:
                if s.recipes_achieved == s2['recipes']:
                    found = True
                    break
            if not found:
                self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
