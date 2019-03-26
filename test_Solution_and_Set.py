import unittest
import numpy as np
import veg_vampire as vv

class Solution_Test(unittest.TestCase):
        def test_constructor(self):
            s = vv.Solution()
            s = vv.Solution([], [(4,3)])

        def test_add_move(self):
            s = vv.Solution()
            s.add_move((4, 3))
            s.add_move((4, 2), 0)
            self.assertTrue(s.recipes_achieved[0] == 1)
            self.assertTrue(len(s.moves) == 2)

        def test_copy(self):
            s = vv.Solution([], [(1, 2)])
            s2 = s.copy()
            s2.add_move((2, 2))
            self.assertTrue(len(s.moves) == 1)
            self.assertTrue(len(s2.moves) == 2)

class Solution_Set_Test(unittest.TestCase):
        def test_exclude_dominated_solutions(self):
                sol_list = [vv.Solution([1, 0, 0, 0, 0], []),
                            vv.Solution([2, 0, 0, 0, 0], []),
                            vv.Solution([0, 1, 0, 0, 0], [])]
                ss = vv.Solution_Set(sol_list)
                self.assertTrue(len(ss.solutions) == 2)

        def test_add_solution_to_set(self):
                sol_list = [vv.Solution([1, 0, 0, 0, 0], []),
                            vv.Solution([2, 0, 0, 0, 0], []),
                            vv.Solution([0, 1, 0, 0, 0], [])]
                ss = vv.Solution_Set(sol_list)
                ss.add_solution_to_set(vv.Solution([0, 1, 0, 0, 0], []))
                ss.add_solution_to_set(vv.Solution([2, 1, 0, 0, 0], []))
                ss.add_solution_to_set(vv.Solution([0, 0, 1, 0, 0], []))
                self.assertTrue(len(ss.solutions) == 2)


        def test_sort_by_total(self):
                sol_list = [vv.Solution([1, 0, 1, 1, 0], []),
                            vv.Solution([2, 0, 0, 0, 0], []),
                            vv.Solution([0, 1, 0, 0, 5], [])]
                ss = vv.Solution_Set(sol_list)
                ss.sort_by_total()
                sums = [sum(s.recipes_achieved) for s in ss]
                self.assertEqual(sums, [6, 3, 2])
        # FIXME: test_verify_solution
                
if __name__ == '__main__':
        unittest.main()
