import unittest
from pathlib import Path
import numpy as np
import copy
import sys
import re

from grid_maker import Map
NUMBER_TYPE = (int, float)

"""
Some magic to import your code (assignment) into this file for testing.
Please do not change the code below.
"""

EXERCISE_OR_ASSIGNMENT_NAME = "assignment2"

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

    def interpolate_nodes(self, node1, node2):
        direction = (node2[0] - node1[0], node2[1] - node1[1])
        direction = (direction[0] / sum(direction), direction[1] / sum(direction))
        self.assertEqual(direction[0], int(direction[0]))
        self.assertEqual(direction[1], int(direction[1]))

        path = [node1]
        node = node1
        for _ in range(abs(node2[0] - node1[0]) + abs(node2[1] - node1[1])):
            node = (int(node[0] + direction[0]), int(node[1] + direction[1]))
            path.append(node)
        return path

    def check_path(self, map, path):
        node = path[0]
        for next_node in path[1:]:
            self.assertFalse(node[0] != next_node[0] and node[1] != next_node[1],
                             msg="nodes in a path should be orthogonal to each other!")
            for coordinate in self.interpolate_nodes(node, next_node):
                self.assertTrue(map[coordinate], msg="Your path does not follow the roads!")
            node = next_node

class TestGraph(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        self.assertIsInstance(graph, Graph)
        self.assertIsInstance(graph.adjacency_list, dict)
        key = list(graph.adjacency_list.keys())[0]
        self.assertIsInstance(key, tuple)
        self.assertIsInstance(key[0], int)
        self.assertIsInstance(key[1], int)
        value = graph.adjacency_list[(0,0)]
        self.assertIsInstance(value, set)
        value = list(value)[0]
        self.assertIsInstance(value, tuple)
        self.assertIsInstance(value[0][0], int)
        self.assertIsInstance(value[0][1], int)
        self.assertIsInstance(value[1], NUMBER_TYPE)
        self.assertIsInstance(value[2], NUMBER_TYPE)

    def test2_has_adjacency_list(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        self.assertFalse({} == graph.adjacency_list, msg="The adjacency list should not be empty!")
        for edges in graph.adjacency_list.values():
            self.assertFalse(set() == edges, msg="All nodes should have atleast one edge!")

    def test3_is_correct(self):
        map_ = Map(0, (0, 0))
        map_.grid = np.ones((3,4)) * 15
        map_.grid[1,1:3] = 0
        graph = Graph(map_)
        edges = {(0,0): {((0,3), 3, 15.0), ((2,0), 2, 15.0)},
                 (2,0): {((0,0), 2, 15.0), ((2,3), 3, 15.0)},
                 (0,3): {((0,0), 3, 15.0), ((2,3), 2, 15.0)},
                 (2,3): {((0,3), 2, 15.0), ((2,0), 3, 15.0)}}

        for node, edge in graph.adjacency_list.items():
            self.assertIn(node, [(0,0), (2,0), (0,3), (2,3)])
            self.assertSetEqual(edges[node], edge)

class TestFloodFillSolver(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver = FloodFillSolver()
        return_value = solver(map_, start, end)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], NUMBER_TYPE)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

    def test2_next_step(self):
        map_ = Map(0, (0, 0))
        solver = FloodFillSolver()
        solver.grid = map_.grid
        end = (map_.shape[0]-1, map_.shape[1]-1)
        self.assertSetEqual(set(solver.next_step((0, 0))), {(0,1), (1,0)})
        self.assertSetEqual(set(solver.next_step(end)), {(map_.shape[0]-1, map_.shape[1]-2),
                                                         (map_.shape[0]-2, map_.shape[1]-1)})

    def test3_find_path(self):
        map_ = Map(0, (0, 0))
        solver = FloodFillSolver()
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver.history = {start: None}
        node = start
        for next_node in self.interpolate_nodes(start, (0, map_.shape[1]-1))[1:] + self.interpolate_nodes((0, map_.shape[1]-1), end)[1:]:
            solver.history[next_node] = node
            node = next_node
        solver.destination = end
        path, length = solver.find_path()
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

    def test4_call(self):
        map_ = Map(0, (0, 0))
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        path, length = FloodFillSolver()(map_, start, end)
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

class TestFloodFillSolverGraph(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver = FloodFillSolverGraph()
        return_value = solver(graph, start, end)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], NUMBER_TYPE)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

    def test2_find_path(self):
        map_ = Map(0, (0, 0))
        solver = FloodFillSolverGraph()
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver.history = {start: None}
        node = start
        for next_node in self.interpolate_nodes(start, (0, map_.shape[1]-1))[1:] + self.interpolate_nodes((0, map_.shape[1]-1), end)[1:]:
            solver.history[next_node] = node
            node = next_node
        solver.destination = end
        path, length = solver.find_path()
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

    def test3_call(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        path, length = FloodFillSolverGraph()(graph, start, end)
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

class TestBFSSolverShortestPath(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver = BFSSolverShortestPath()
        return_value = solver(graph, start, end)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], NUMBER_TYPE)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

    def test2_find_path(self):
        map_ = Map(0, (0, 0))
        solver = BFSSolverShortestPath()
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver.history = {start: (None, 0),
                          (0, map_.shape[1]-1): (start, map_.shape[1]-1),
                          end: ((0, map_.shape[1]-1), map_.shape[0] + map_.shape[1] - 2)}
        solver.destination = end
        path, length = solver.find_path()
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

    def test3_call(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        path, length = BFSSolverShortestPath()(graph, start, end)
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

class TestBFSSolverFastestPath(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver = BFSSolverFastestPath()
        return_value = solver(graph, start, end, 100)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], float)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

    def test2_find_path(self):
        map_ = Map(0, (0, 0))
        solver = BFSSolverFastestPath()
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        solver.history = {start: (None, 0),
                          (0, map_.shape[1]-1): (start, map_.shape[1]-1),
                          end: ((0, map_.shape[1]-1), (map_.shape[0] + map_.shape[1] - 2))}
        solver.destination = end
        path, length = solver.find_path()
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual(map_.shape[0] + map_.shape[1] - 2, length)
        self.check_path(map_, path)

    def test3_call(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0, 0)
        end = (map_.shape[0]-1, map_.shape[1]-1)
        path, length = BFSSolverFastestPath()(graph, start, end, 100)
        self.assertEqual(start, path[0])
        self.assertEqual(end, path[-1])
        self.assertAlmostEqual((map_.shape[0] + map_.shape[1] - 2)/30, length)
        self.check_path(map_, path)

class TestCoordinate_to_node(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        return_value = coordinate_to_node(map_, graph, (0,0))
        self.assertIsInstance(return_value, list)
        self.assertIsInstance(return_value[0], tuple)
        self.assertIsInstance(return_value[0][0], int)
        self.assertIsInstance(return_value[0][1], int)

    def test2_is_node(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        return_value = coordinate_to_node(map_, graph, (0,0))
        self.assertEqual(1, len(return_value))
        self.assertTupleEqual((0,0), return_value[0])

    def test2_is_not_node(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        return_value = coordinate_to_node(map_, graph, (0,1))
        self.assertEqual(2, len(return_value))
        self.assertCountEqual([(0, 0), (0, map_.shape[1]-1)], return_value)

class TestBFSSolverMultipleFastestPaths(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = [((0,0), 0)]
        end = [((map_.shape[0]-1, map_.shape[1]-1), 0)]
        solver = BFSSolverMultipleFastestPaths()
        return_value = solver(graph, start, end, 100)
        self.assertIsInstance(return_value, list)
        return_value = return_value[0]
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], float)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

class TestBFSSolverFastestPathMD(ExtendTestCase):
    def test1_types(self):
        map_ = Map(0, (0, 0))
        graph = Graph(map_)
        start = (0,0)
        end = [(map_.shape[0]-1, map_.shape[1]-1)]
        solver = BFSSolverFastestPathMD()
        return_value = solver(graph, start, end, 100)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], float)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

class TestFind_path(ExtendTestCase):
    def test1_types(self):
        map_ = Map(5, (1, 1))
        start = map_.get_coordinate_in_city(RNG.integers(len(map_.city_grids)-1))
        end = map_.get_coordinate_in_city(RNG.integers(len(map_.city_grids)-1))
        return_value = find_path(start, end, map_, 100)
        self.assertIsInstance(return_value, tuple)
        self.assertIsInstance(return_value[0], list)
        self.assertIsInstance(return_value[1], float)
        self.assertIsInstance(return_value[0][0], tuple)
        self.assertIsInstance(return_value[0][0][0], int)
        self.assertIsInstance(return_value[0][0][1], int)

# run the unittest if the file is just run as a normal python file (without extra command line options)
if __name__ == "__main__" and Path.cwd() / sys.argv[0] == Path.cwd() / __file__:
    # Run the tests
    for tests in [obj for obj in dir() if obj[:4] == "Test"]:
        suite = unittest.TestLoader().loadTestsFromTestCase(locals()[tests])
        unittest.TextTestRunner(verbosity=2).run(suite)

