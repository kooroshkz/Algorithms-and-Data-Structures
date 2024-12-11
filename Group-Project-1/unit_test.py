import unittest
from pathlib import Path
import numpy as np
import copy
import sys
import re

"""
Some magic to import your code (assignment) into this file for testing.
Please do not change the code below.
"""

EXERCISE_OR_ASSIGNMENT_NAME = "assignment"

# import any student file, when running from the command line without any flags
if Path.cwd() / sys.argv[0] == Path.cwd() / __file__:
    student_file = Path.cwd()
    student_file = next(student_file.glob(f'../**/{EXERCISE_OR_ASSIGNMENT_NAME}*[!_backup|_notebook].py'))
# import any student file, when running from the command using unittest flags
elif re.fullmatch(r'python3? -m unittest', sys.argv[0], re.IGNORECASE):
    student_file = Path.cwd()
    student_file = next(student_file.glob(f'../**/{EXERCISE_OR_ASSIGNMENT_NAME}*[!_backup|_notebook].py'))
# import the student file that imported the unit_test
elif (Path.cwd() / sys.argv[0]).parent == (Path.cwd() / __file__).parent:
    student_file = Path(sys.argv[0])
# import any student file, when running using PyCharm or Vscode
else:
    student_file = Path.cwd()
    student_file = next(student_file.glob(f'../**/{EXERCISE_OR_ASSIGNMENT_NAME}*[!_backup|_notebook].py'))
sys.path.append(str(student_file.parent))

# import student code
m = __import__(student_file.stem)

# find all imports, either with __all__ or dir
try:
    attrlist = m.__all__
except AttributeError:
    attrlist = dir(m)

# add all student code to this namespace.
for attr in attrlist:
    if attr[:2] != "__":
        globals()[attr] = getattr(m, attr)

"""
DO NOT CHANGE THE CODE BELOW!
THESE TEST ARE VERY BASIC TEST TO GIVE AN IDEA IF YOU ARE ON THE RIGHT TRACK!
"""


class ExtendTestCase(unittest.TestCase):
    def assertArrayEqual(self, in_, out):
        self.assertIsInstance(out, np.ndarray, f"Expected numpy array.")
        self.assertEqual(in_.shape, out.shape, f"Expected {in_.shape} got {out.shape}.")
        equal = np.isclose(in_, out)
        self.assertTrue(equal.all(), f"Expected {in_} got {out}.")

class TestSudoku(ExtendTestCase):
    def test_attributes(self):
        s = Sudoku()
        self.assertCountEqual(["grid", "size"],
                              vars(s).keys(),
                              msg=f"The class Sudoku does not have the right attributes!")

    def test_methods(self):
        methods = ['set_grid', 'get_row', 'get_col', 'get_box_index', 'get_box', 'is_set_correct',
                   'check_cell', 'check_sudoku', 'step', 'next_step', 'clean_up', 'solve']
        self.assertCountEqual(methods,
                              filter(lambda x: x[0] != "_",  vars(Sudoku)),
                              msg=f"The class Sudoku does not have the right methods!")

    def test_set_grid(self):
        s = Sudoku()
        for i in [4, 9, 16]:
            in_ = RNG.integers(1, i+1, size=(i,i))
            s.set_grid(in_)
            self.assertArrayEqual(in_, s.grid)
            self.assertEqual(i, s.size)

    def test_get_row(self):
        s = Sudoku()
        in_ = RNG.integers(1, 10, size=(9, 9))
        s.grid = in_
        for i in range(9):self.assertArrayEqual(in_[i], s.get_row(i))

    def test_get_col(self):
        s = Sudoku()
        in_ = RNG.integers(1, 10, size=(9, 9))
        s.grid = in_
        for i in range(9):self.assertArrayEqual(in_[:,i], s.get_col(i))

    def test_box_index(self):
        s = Sudoku()
        for i in range(9):
            self.assertEqual(i, s.get_box_index(int(np.ceil((i+1)/3)*3-3+(i%3)),(i%3)*3+i//3))

    def test_get_box(self):
        s = Sudoku()
        in_ = RNG.integers(1, 10, size=(9, 9))
        s.grid = in_
        for i in range(9):
            self.assertArrayEqual(in_[(a:=np.floor(i/3).astype(int))*3:(a+1)*3,(b:=(i%3)*3):b+3], s.get_box(i))

    def test_is_set_correct(self):
        self.assertTrue(Sudoku.is_set_correct(np.array([0,0,1,0,2,3,7,4,9])))
        self.assertTrue(Sudoku.is_set_correct(np.array([6,8,1,5,2,3,7,4,9])))
        self.assertFalse(Sudoku.is_set_correct(np.array([6,6,1,5,2,3,7,4,9])))
        self.assertFalse(Sudoku.is_set_correct(np.array([0,0,1,0,2,3,7,1,9])))

    def test_check_sudoku(self):
        self.assertTrue(ftos("small_test.txt").check_sudoku())
        self.assertTrue(ftos("medium_test.txt").check_sudoku())
        self.assertTrue(ftos("large_test.txt").check_sudoku())

    def test_solve_exhaustive_search(self):
        self.assertTrue(ftos("small_test.txt").solve(False))

    def test_solve_backtracking(self):
        self.assertTrue(ftos("small_test.txt").solve(True))
        self.assertTrue(ftos("medium_test.txt").solve(True))
        self.assertTrue(ftos("large_test.txt").solve(True))

def ftos(filename):
    sudoku = Sudoku()
    with open(filename, 'r') as f:
        sudoku.grid = np.array([list(map(int, line.split(','))) for line in f])
        sudoku.size = int(np.sqrt(sudoku.grid.size))
    return sudoku
