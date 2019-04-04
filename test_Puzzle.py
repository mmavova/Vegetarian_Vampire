import unittest
import numpy as np
import veg_vampire as vv


class Puzzle_Test(unittest.TestCase):
    def test_constructor(self):
        b = np.array([[1, 0, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 0, 1, 0, 1]])
        p = vv.Puzzle(b)
        p2 = vv.Puzzle.from_puzzle(p)

    def test_verify_initial_state_correctedness(self):
        b = np.array([[1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1]])
        p = vv.Puzzle(b)
        p.verify_initial_state_correctedness()

        b = np.array([[1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 0, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1]])
        with self.assertRaises(Exception) as cm:
            p = vv.Puzzle(b, [0] * 5)
            p.verify_initial_state_correctedness()
        err = cm.exception
        self.assertEqual(str(err), "The starting number of plants must be divisible by 5.\n"
                         "Check both the board and the buffer.")

        with self.assertRaises(Exception) as cm:
            p = vv.Puzzle(b, [1] * 5)
            p.verify_initial_state_correctedness()
        err = cm.exception
        self.assertEqual(str(err), "Buffer cannot be full at the beginning of the puzzle.")

        with self.assertRaises(Exception) as cm:
            p = vv.Puzzle(b, [0, 0, 0, 1, 0])
            p.verify_initial_state_correctedness()
        err = cm.exception
        self.assertEqual(str(err), "Buffer has to be filled from left to right without any gaps")

    def test_is_square_reachable(self):
        p = vv.Puzzle(np.array([[1, 0, 1, 0, 1],
                                [1, 1, 1, 0, 1],
                                [1, 1, 1, 0, 1],
                                [1, 1, 1, 0, 1],
                                [1, 0, 1, 0, 1]]))
        self.assertTrue(p.is_square_reachable(1, 1))
        self.assertTrue(p.is_square_reachable(0, 0))
        self.assertTrue(p.is_square_reachable(1, 1))
        self.assertTrue(p.is_square_reachable(4, 3))
        self.assertTrue(p.is_square_reachable(2, 2))
        self.assertFalse(p.is_square_reachable(2, 1))
        self.assertFalse(p.is_square_reachable(2, 0))
        p = vv.Puzzle(np.array([[1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 1]]))
        self.assertTrue(p.is_square_reachable(0, 1))
        self.assertTrue(p.is_square_reachable(1, 4))
        self.assertTrue(p.is_square_reachable(2, 4))
        self.assertTrue(p.is_square_reachable(3, 4))
        self.assertFalse(p.is_square_reachable(4, 4))
        self.assertFalse(p.is_square_reachable(2, 2))

    def test_is_every_square_reachable(self):
        p = vv.Puzzle(np.array([[1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 1]]))
        self.assertFalse(p.is_every_square_reachable())
        p = vv.Puzzle(np.array([[1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 0],
                                [1, 1, 0, 0, 1],
                                [1, 0, 1, 1, 0],
                                [1, 1, 1, 1, 1]]))
        self.assertTrue(p.is_every_square_reachable())

        p = vv.Puzzle(np.array([[1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 0],
                                [1, 1, 0, 0, 1],
                                [1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 1]]))
        self.assertFalse(p.is_every_square_reachable())

    def test_buf_begins_recipe(self):
        b = np.zeros([5, 5], dtype=int)
        p = vv.Puzzle(b, [1, 0, 0, 0, 0])
        n, the_list = p.buf_begins_recipe()
        self.assertTrue(n == 1)
        self.assertTrue([0] == the_list)
        p = vv.Puzzle(b, [3, 2, 2, 3, 4])
        n, the_list = p.buf_begins_recipe()
        self.assertTrue(n == 5)
        self.assertTrue([4] == the_list)
        p = vv.Puzzle(b, [3, 0, 0, 0, 0])
        n, the_list = p.buf_begins_recipe()
        self.assertTrue(n == 1)
        self.assertTrue([3, 4] == the_list)


if __name__ == '__main__':
    unittest.main()
