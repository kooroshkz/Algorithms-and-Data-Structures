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

EXERCISE_OR_ASSIGNMENT_NAME = "exercise5"

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

RNG = np.random.default_rng()

"""
DO NOT CHANGE THE CODE BELOW!
THESE TEST ARE VERY BASIC TEST TO GIVE AN IDEA IF YOU ARE ON THE RIGHT TRACK!
"""

class TestTree(unittest.TestCase):
    def test_question1(self):
        distance([3,4],[0,0])
        n_points = RNG.integers(10, 20)
        x, y = (RNG.integers(10), RNG.integers(10))
        data = [np.array(list(zip(RNG.normal(x, 1, n_points), RNG.normal(y, 1, n_points)))) for _ in range(RNG.integers(3, 6))]
        nearest_neighbour(data[0], [x, y])
        classify_point(data, [x, y])

    def test_question2(self):
        door = DoorRingPuzzle()
        SolveDoorRingPuzzle()(door)

    def test_question3(self):
        tree = TernaryTree()
        tree.add(10)
        tree.search(10)

    def test_question4(self):
        graph = CompleteGraph(3)
        cycle = list(RNG.permutation(3))
        length_of_circuit(graph, cycle)
        find_shortest_circuit(graph)


if __name__ == "__main__":
    for tests in [obj for obj in dir() if obj[:4] == "Test"]:
        suite = unittest.TestLoader().loadTestsFromTestCase(locals()[tests])
        unittest.TextTestRunner(verbosity=2).run(suite)