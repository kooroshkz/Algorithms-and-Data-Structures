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

EXERCISE_OR_ASSIGNMENT_NAME = "exercise12"

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

class TestLab(unittest.TestCase):
    def test_question1(self):
        graph = Graph()
        graph.generate_random_graph()
        dijkstra = Dijkstra()
        out = dijkstra(graph, graph.get_random_node(), False)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], tuple)
        self.assertIsInstance(out[0][0], (int, np.int64, np.int32))

        self.assertIsInstance(dijkstra.find_shortest_edges(), list)
        self.assertIsInstance(dijkstra.find_shortest_edges()[0], tuple)
        self.assertIsInstance(dijkstra.find_shortest_edges()[0][0], (int, np.int64, np.int32))

    def test_question2(self):
        graph = Graph()
        graph.generate_random_graph()
        prim = Prim()
        out = prim(graph, False)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], tuple)
        self.assertIsInstance(out[0][0], (int, np.int64, np.int32))

    def test_question3(self):
        graph = Graph()
        graph.generate_random_graph()
        kruskal = Kruskal()
        out = kruskal(graph, False)
        self.assertIsInstance(kruskal.edges, list)
        self.assertIsInstance(kruskal.forest, Forest)
        self.assertIsInstance(kruskal.queue, list)

        self.assertIsInstance(kruskal.create_forest(), Forest)

        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], tuple)
        self.assertIsInstance(out[0][0], (int, np.int64, np.int32))

        forest = Forest(10)
        self.assertIsInstance(forest.trees, list)
        self.assertIsInstance(forest.trees[0], set)
        self.assertIsInstance(forest.find_tree(2), int)

        forest = ForestFast(10)
        self.assertIsInstance(forest.trees, np.ndarray)
        self.assertIsInstance(forest.trees[0], (np.int64, np.int32))
        self.assertIsInstance(forest.find_tree(2), int)

        kruskal = KruskalFast()
        out = kruskal(graph, False)
        self.assertIsInstance(kruskal.create_forest(), ForestFast)

        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], tuple)
        self.assertIsInstance(out[0][0], (int, np.int64, np.int32))

if __name__ == "__main__":
    for tests in [obj for obj in dir() if obj[:4] == "Test"]:
        suite = unittest.TestLoader().loadTestsFromTestCase(locals()[tests])
        unittest.TextTestRunner(verbosity=2).run(suite)
