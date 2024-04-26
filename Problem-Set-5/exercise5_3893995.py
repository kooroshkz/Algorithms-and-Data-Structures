############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from tree import hierarchy_pos

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
def distance(pointA, pointB):
    """
    This calculates the Euclidean distance between two points: https://en.wikipedia.org/wiki/Euclidean_distance

    :param pointA: The first coordinate
    :type pointA: list[float] or np.ndarray[(2,), float]
    :param pointB: The second coordinate
    :type pointB: list[float] or np.ndarray[(2,), float]
    :return: The distance between the two points
    :rtype: float
    """
    return np.linalg.norm(np.array(pointA) - np.array(pointB))

def nearest_neighbour(data, point):
    """
    This function finds the nearest neighbour of "point" in the "data".

    :param data: All the points (neighbours) that need to be compared to "point".
    :type data: np.ndarray[(n, 2), float]
    :param point: The point of which you want to find the closest neighbour.
    :type point: list[float] or np.ndarray[(2,), float]
    :return: The nearest neighbour and the distance to that neighbour.
    :rtype: tuple[np.ndarray[(2,), float], float]
    """
    return min([(neighbour, distance(neighbour, point)) for neighbour in data if not np.array_equal(neighbour, point)], key=lambda x: x[1])

############ CODE BLOCK 15 ################
def classify_point(data, point):
    """
    This function finds the nearest group based on the nearest neighbour of each group.
    Use `nearest_neighbour` in this function.

    :param data: A list of groups, where each group consists of all the points (neighbours) that need to be compared to "point".
    :type data: list[np.ndarray[(Any, 2), float]]
    :param point: The point of which you want to find the closest group.
    :type point: list[float] or np.ndarray[(2,), float]
    :return: The nearest group and the nearest neighbour.
    :rtype: tuple[int, np.ndarray[(Any, 2), float]]
    """
    return min([(i, nearest_neighbour(group, point)) for i, group in enumerate(data)], key=lambda x: x[1][1])

############ CODE BLOCK 20 ################
class Ring():
    """
    One ring of the door.
    """
    def __init__(self, current=0, size=3):
        """
        Attributes:
            :param current: Current value that is on top of the ring.
            :type current: int
            :param size: The number of options that this ring has.
            :type size: int
            :param answer: the correct position of this ring.
            :type answer: int            
        """
        self.__current = current
        self.__size = size
        self.__answer = RNG.integers(size)

    def turn(self):
        """
        This method turns the ring clockwise and
        returns if the ring is in the original order.

        :return: if the ring is in the original order.
        :rtype: boolean
        """
        self.__current = (self.__current + 1) % self.__size
        return not self.__current

    def correct(self):
        """
        This method check if the ring is currently in the right position.
        """
        return self.__current == self.__answer
        
    def __repr__(self):
        return f"Ring({chr(self.__current + 65)})"

class DoorRingPuzzle():
    """
    This class represents a door with a certain amount of rings 
    that need to be positioned in the correct order to open the door.
    """
    def __init__(self, n_rings=None, size=3):
        """
        This initialized the door. 
        
        Attributes:
            :param rings: The rings of the door.
            :type rings: list[Ring}


        :param n_rings: The number of rings this door has, defaults to 3
        :type n_rings: int, optional
        :param size: The size of each ring (number of options), defaults to 3
                     This can also be a list with the size of each individual ring.
                     This list should have the same length as n_rings.
        :type size: list[int] or int, optional
        """
        if not isinstance(size, int):
            if n_rings != len(size) and not n_rings is None:
                raise ValueError("The number of rings should be equal to the number of sizes that are given for each individual ring!")
            self.__rings = [Ring(0, s) for s in size]            
        else:
            if n_rings is None:
                n_rings = 3
            self.__rings = [Ring(0, size) for _ in range(n_rings)]

    def turn_ring(self, ring):
        """
        This method can rotate one ring clockwise.
        It also tells the user if the ring is back in its original position.
        Thus with the "A" on top.

        :param ring: The ring that is rotated.
        :type ring: int
        :return: If the ring is in its original position.
        :rtype: boolean
        """
        return self.__rings[ring].turn()

    def open_door(self):
        """
        This method checks if you can open the door.

        :return: If opening the door succeeded.
        :rtype: boolean
        """
        for ring in self.__rings:
            if not ring.correct():
                return False
        return True

    def __len__(self):
        """
        This gives the length of the door which
        is defined as the number of rings.
        """
        return len(self.__rings)

    def __repr__(self):
        return str(self.__rings)    

############ CODE BLOCK 25 ################
class SolveDoorRingPuzzle():
    def __call__(self, door):
        """
        This method solves the door with "n" rings problem.
        You do not need to return anything because of side effects.
        See, this exercise of ITP on an explanation of side effects:
        https://joshhug.github.io/LeidenITP/labs/lab7/#exercise-1-shopping-list-standard

        :param door: The door that needs to be opened.
        :type door: DoorRingPuzzle
        """
        self.door = door
    
    def step(self, ring):
        """
        This is one step in the exhaustive search algorithm which uses depth-first search.

        :param ring: The ring that is currently turned.
        :type ring: int
        :return: If the door is opened in the current configuration.
        :rtype: boolean
        """
        if ring == len(self.door):
            return self.door.open_door()
        if self.door.turn_ring(ring):
            return self.step(ring + 1)
        return self.step(ring)
            
    def next_step(self, ring):
        """
        This determines the next step in the exhaustive search.
    
        :param ring: The ring that is currently turned.
        :type ring: int
        :return: This method returns what self.step returns
        :type: boolean
        """
        return self.step(ring)

############ CODE BLOCK 30 ################
class Node():
    def __init__(self, value, left=None, middle=None, right=None):
        """
        This is a node for a ternary tree.

        Attributes:
            :param info: The value of the node.
            :type: info: int
            :param left: The left child of the node, defaults to None
            :type left: Node, optional
            :param middle: The left child of the node, defaults to None
            :type middle: Node, optional
            :param right: The left child of the node, defaults to None
            :type right: Node, optional
        """
        self.info = value
        self.left = left
        self.middle = middle
        self.right = right

    def __repr__(self):
        return f"Node({self.info}) -> {self.left.info if self.left is not None else 'None', self.middle.info if self.middle is not None else 'None', self.right.info if self.right is not None else 'None'}"

class TernaryTree():
    def __init__(self):
        """
        This initializes the tree which is always initialized as an empty tree.

        Attributes:
            :param root: The root of the tree
            :type root: Node
        """
        self.root = None

    def add(self, value):
        """
        Randomly add values to the tree.
        You could do this by randomly traversing the tree and 
        add a new node when a empty leaf node is found.

        :param value: The value that is added to the tree
        :type value: int       
        """
        if self.root is None:
            self.root = Node(value)
        else:
            self.__add(value, self.root)  

    def show(self):
        """
        This method shows the tree, where the root node is colored blue, 
        the left nodes are colored green, and the right nodes are colored red.
        """
        if self.root is None:
            raise ValueError("This is an empty tree and can not be show.")
            
        # Recursively add all edges and nodes.
        def add_node_edge(G, color_map, parent_graph_node, node):
            # In case of printing a binary tree check if a node exists
            if node.info in G:
                i = 2
                while f"{node.info}_{i}" in G:
                    i += 1
                node_name = f"{node.info}_{i}"
            else:
                node_name = node.info
            G.add_node(node_name)

            # Make root node or edge to parent node
            if parent_graph_node is not None:
                G.add_edge(parent_graph_node, node_name)
            
            if node.left is not None:
                add_node_edge(G, color_map, node_name, node.left)
            if node.middle is not None:
                add_node_edge(G, color_map, node_name, node.middle)
            if node.right is not None:
                add_node_edge(G, color_map, node_name, node.right)
        
        # Make the graph
        G = nx.DiGraph()
        color_map = []
        add_node_edge(G, color_map, None, self.root)
        name_root = self.root.info

        # Generate the node positions
        pos = hierarchy_pos(G, root=self.root.info, leaf_vs_root_factor=1)
        new_pos = {k:v for k,v in pos.items() if str(k)[0] != 'N'}
        k = G.subgraph(new_pos.keys())

        nx.draw(k, pos=new_pos, with_labels=True, node_size=1000)

        # Set the plot settings
        x, y = zip(*pos.values())
        x_min, x_max = min(x), max(x)
        plt.xlim(1.015*x_min-0.015*x_max, 1.015*x_max-0.015*x_min)
        plt.ylim(min(y)-0.08, max(y)+0.08)
        plt.show()

############ CODE BLOCK 35 ################
    def search(self, value):
        """
        This method search for a node with the value "value".
        If the node is not found it returns None.

        :param value: The value that is search for.
        :type value: int
        :return: This returns the node
        :rtype: Node
        """
        if self.root is None:
            return None
        return self._step(self.root, value)
        
    @staticmethod
    def _step(current_node, value):
        """
        This is a recursive helper method for "search".
        This makes it possible to do the exhaustive search.
        
        :param value: The value that is search for.
        :type value: int
        :param current_node: The current node in the Ternary tree.
        :type current_node: Node
        :return: This returns the node
        :rtype: Node
        """
        if current_node.info == value:
            return current_node
        if current_node.left is not None:
            result = TernaryTree._step(current_node.left, value)
            if result is not None:
                return result
        if current_node.middle is not None:
            result = TernaryTree._step(current_node.middle, value)
            if result is not None:
                return result
        if current_node.right is not None:
            result = TernaryTree._step(current_node.right, value)
            if result is not None:
                return result
        return None
    

############ CODE BLOCK 40 ################
class CompleteGraph():
    def __init__(self, size=5):
        """
        This initializes a complete graph with a certain size.
        The internal representation is an adjacency matrix.

        :param size: The size of the graph, i.e., the number of nodes.
        :type size: int
        """
        self.adjacency_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i):
                self.adjacency_matrix[i, j] = self.adjacency_matrix[j, i] = RNG.integers(1, 10)

    def __getitem__(self, index):
        """
        This makes the graph indexable as if you are directly indexing "self.adjacency_matrix"
        """
        return self.adjacency_matrix[index]

    def __repr__(self):
        return repr(self.adjacency_matrix)
            
    def show(self):
        """
        This method shows the current graph.
        """
        graph = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.shell_layout(graph)
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx(graph, 
                         pos,
                         with_labels=True,
                         node_size=400,
                         width=2,
                         arrowsize=15)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=0.15, font_size=10)
        plt.show()

def find_shortest_circuit(graph):
    """
    This function finds the shortest Hamiltonian circuit in a graph and
    returns the path and the length of this path.

    :param graph: The graph
    :type graph: CompleteGraph
    :return: The shortest circuit and its cost
    :rtype: tuple[tuple[int], int]
    """
    # Define a helper function to generate all possible permutations of nodes
    def generate_permutations(nodes):
        if len(nodes) <= 1:
            yield nodes
        else:
            for i in range(len(nodes)):
                for perm in generate_permutations(nodes[:i] + nodes[i+1:]):
                    yield (nodes[i],) + tuple(perm)

    min_cost = float('inf')
    shortest_circuit = None

    # Generate all possible Hamiltonian circuits
    for perm in generate_permutations(list(range(len(graph.adjacency_matrix)))):
        # Ensure that the circuit is closed
        perm += (perm[0],)
        cost = length_of_circuit(graph, perm)
        if cost < min_cost:
            min_cost = cost
            shortest_circuit = perm

    return shortest_circuit, min_cost


def length_of_circuit(graph, cycle):
    """
    This is a helper function to calculate the length of one circuit.

    :param graph: The graph which the cycle traverses
    :type graph: CompleteGraph
    :param cycle: The cycle that encodes the circuit. 
                  This should be just a list of nodes.
    :type cycle: list[int]
    :return: The length of this circuit
    :rtype: int
    """
    # Sum up the weights of the edges in the cycle
    total_length = sum(graph.adjacency_matrix[cycle[i], cycle[i + 1]] for i in range(len(cycle) - 1))
    return total_length


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
