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

EXERCISE_OR_ASSIGNMENT_NAME = "exercise1"

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

"""
DO NOT CHANGE THE CODE BELOW!
THESE TEST ARE VERY BASIC TEST TO GIVE AN IDEA IF YOU ARE ON THE RIGHT TRACK!
"""

class TestTree(unittest.TestCase):
    def test_attributes(self):
        g = Graph()
        self.assertCountEqual(["adjacency_list"],
                              vars(g).keys(),
                              msg=f"The class Node does not have the right attributes!")

    def test_methods(self):
        g = Graph()
        methods = ['add_edge', 'add_vertex', 'adjacency_list', 'get_adjacency_list',
                   'get_adjacency_matrix', 'has_two_or_less_odd_vertices',
                   'invert_edges', 'list_to_matrix', 'matrix_to_list',
                   'set_adjacency_list', 'set_adjacency_matrix', 'show',
                   'to_undirected_graph']
        self.assertCountEqual(methods,
                              filter(lambda x: x[0] != "_",  dir(g)),
                              msg=f"The class Sudoku does not have the right methods!")

# run the unittest if the file is just run as a normal python file (without extra command line options)
if __name__ == "__main__" and Path.cwd() / sys.argv[0] == Path.cwd() / __file__:
    # Run the tests
    for tests in [obj for obj in dir() if obj[:4] == "Test"]:
        suite = unittest.TestLoader().loadTestsFromTestCase(locals()[tests])
        unittest.TextTestRunner(verbosity=2).run(suite)