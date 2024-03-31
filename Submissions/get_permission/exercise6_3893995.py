############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
class Permutations():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating all state-space but 
        using backtracking to ignore incorrect solution/solution branches.
        
        :param list_: The list containing all the unique elements.
        :type list_: list
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # All actions are the same, so it is helpful to make an object attribute.
        self.list = list_

        return self.step([])

    def step(self, chosen_list):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.
        
        :param chosen_list: This list contains all objects chosen so far.
        :type chosen_list: list[Objects]
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        # If the permutation is correct, we can return it.
        if len(chosen_list) == len(self.list):
            return [chosen_list]

        # If the permutation is incorrect, we can return an empty list.
        if not self.is_not_incorrect(chosen_list):
            return []

        # If the permutation is not incorrect, we can generate the next step.
        return self.next_step(chosen_list, self.list)
        
    def next_step(self, chosen_list, chosen_object):
        """
        This method generates the actions that are possible for the next step and calls step with the updated state-space.
        These actions consist of all elements of the original list. 
        Note, that if you saved them as an object attribute this method does not do much.

        :param i: A counter how many elements are added to the new permutation.
        :type i: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        # We need to make a copy of the list, so we can remove the object from the list.
        chosen_object = copy.deepcopy(chosen_object)
        permutations = []
        
        # We iterate over all objects in the list.
        for obj in chosen_object:
            # We remove the object from the list.
            chosen_object.remove(obj)
            # We add the object to the chosen list.
            new_chosen_list = copy.deepcopy(chosen_list)
            new_chosen_list.append(obj)
            # We call the next step with the new state-space.
            permutations += self.step(new_chosen_list)
            # We add the object back to the list.
            chosen_object.append(obj)
        
        return permutations
    
    def is_not_incorrect(self, chosen_list):
        """
        This method returns if the state-space is (partially) correct aka if it can become a permutation.

        :param chosen_list: A possible permutation of self.list
        :type chosen_list: list[Objects]
        :return: Return if the permutation variable is or is not a permutation.
        :rtype: boolean
        """
        # We check if the list is a permutation.
        return len(set(chosen_list)) == len(chosen_list)

############ CODE BLOCK 20 ################
def constraint(queens, col):
    """
    The constraints for the n-queen problem

    :param queens: The currently placed queens.
    :type queens: list[int]
    :param col: The column that the queen is placed in.
    :type col: int
    :return: If the constraint is satisfied or not.
    :rtype: bool
    """
    row = len(queens)
    for r, c in enumerate(queens):
        if c == col or r - c == row - col or r + c == row + col:
            return False
    return True
       

def rec_nQueens(size, queens=None):
    """
    Recursively computes a solution for the n-Queens puzzle.

    :param size: The size of the puzzle
    :type size: int
    :param queens: The currently placed queens, e.g. [4,2] represent 
                   that on row 0 we placed a queen in the 4th position, 
                   and on row 1 we placed a queen in the 2nd position.
                   This defaults to [].
    :type queens: list[int], optional
    :return: The (partial) list of queen positions.
    :rtype: list[int]
    """
    if queens is None:
        queens = []
    
    if size <= len(queens):
        return queens
    
    for col in range(size):
        if constraint(queens, col):
            queens.append(col)
            candidate_sol = rec_nQueens(size, queens)
            if candidate_sol:
                return candidate_sol
            queens.pop()
    return False

class N_Queens():
    def __call__(self, size):
        """
        Recursively computes a solution for the n-Queens puzzle.
        
        size is not a recursive part of n-queens.
        So, we can store it in an object attribute.
        
        :param size:  The size of the puzzle
        :type size: int
        """
        self.size = size 
        return self.step([])

    def step(self, queens):
        """
        One step in solving the n-queens problem

        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :return: The (partial) list of queen positions.
        :rtype: list[int]
        """
        if self.size <= len(queens):
            return queens
        
        for col in range(self.size):
            candidate_sol = self.next_step(queens, col)
            if candidate_sol:
                return candidate_sol
                
        self.clean_up(queens)
        return False
        
    def next_step(self, queens, col):
        """
        Check if you can go to the next step.
        
        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :param col: The column that is tried to be added.
        :type col: int
        :return: The return of the step method
        :rtype: list[int]
        """
        if self.constraint(queens, col):
            queens.append(col)
            return self.step(queens)
        return False

    def clean_up(self, queens):
        """
        Clean up your previous division, in this case, remove the last queen of the board.
        """
        queens.pop()
    
    def constraint(self, queens, col):
        """
        Check if the constraints are satisfied.

        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :param col: The column that is tried to be added.
        :type col: int
        :return: The return of the step method
        :rtype: list[int]
        """
        row = len(queens)
        for r, c in enumerate(queens):
            if c == col or r - c == row - col or r + c == row + col:
                return False
        return True 

############ CODE BLOCK 25 ################
class N_rooks(N_Queens):
    def constraint(self, rooks, col):
        """
        Check if the constraints are satisfied.

        :param rooks: The currently placed rooks, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type rooks: list[int]
        :param col: The column that is tried to be added.
        :type col: int
        :return: The return of the step method
        :rtype: list[int]
        """
        return col not in rooks  

############ CODE BLOCK 28 ################
class N_Queens_All(N_Queens):        
    def step(self, queens):
        """
        One step in solving the n-queens problem

        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :return: The (partial) list of queen positions.
        :rtype: list[int]
        """
        if self.size <= len(queens):
            return [queens]
        
        solutions = []
        for col in range(self.size):
            candidate_sol = self.next_step(queens, col)
            if candidate_sol:
                solutions += candidate_sol
                
        self.clean_up(queens)
        return solutions

############ CODE BLOCK 30 ################
class Graph():
    """
    An undirected graph (not permitting loop) class with the ability to color the nodes using three colors ('r', 'g', or 'b'). 
    Two nodes that are connected through an edge can not have the same colors.

    Class attributes:
        :param colors: The possible colors to color the graph ('r', 'g', or 'b'). 
                       Note, you must use these strings otherwise the `show` method does not work.
        :type colors: list[str]
    
    Object attributes:
        :param adjacency_list: The representation of the graph.
        :type adjacency_list: dict[str/int, set[str/int]]
        :param color_list: A dictionary with as keys the nodes and as values the colors (R, G, B)
        :type color_list: dict[str/int, str]
    """
    colors = ['r', 'b', 'g']
    
    def __init__(self):
        """
        The graph is always initialized empty, use the `set_graph` method to fill it.
        """
        self.adjacency_list = {}
        self.color_list = {}

    @staticmethod
    def generate_random_graph():
        """
        This is a helper method to generate a random graph
        
        :return: This returns a random adjacency_list
        :rtype: dict[str/int, set[str/int]]
        """
        size = RNG.integers(3, 9)
        
        # add nodes
        adjacency_list = {i if RNG.choice([True, False]) else str(i): set() for i in range(size)}
        
        # Add random directed edges
        for node in adjacency_list.keys():
            adjacency_list[node] = set(i if i in adjacency_list else str(i) for i in RNG.choice(list(range(size)), size=RNG.integers(0.9*size)) if i not in [int(node), str(node)])
        # make the edges undirected
        for source, destinations in adjacency_list.items():
            for destination in destinations:
                adjacency_list[destination].add(source)
            
        return adjacency_list       

    def set_graph(self, adjacency_list):
        """
        This method sets the graph using as input an adjacency list.

        Hint: You need to change both the `adjacency_list` and the `color_list`. 
              Colors are by default None.

        :param adjacency_list: The representation of the graph.
        :type adjacency_list: dict[str/int, set[str/int]]
        """
        self.adjacency_list = adjacency_list
        self.color_list = {node: None for node in adjacency_list.keys()}
        
    def show(self):
        """
        This method shows the current graph.
        """
        n_vertices = len(self.adjacency_list)
        matrix = np.zeros((n_vertices, n_vertices))
        key_to_index = dict(zip(self.adjacency_list.keys(), range(n_vertices)))
        for vertex, edges in self.adjacency_list.items():
            for edge in edges:
                matrix[key_to_index[vertex], key_to_index[edge]] = 1
        
        graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        nx.draw_shell(graph,
                      labels=dict(enumerate(self.adjacency_list.keys())),
                      with_labels=True,
                      node_size=500,
                      width=2,
                      arrowsize=1,
                      node_color=[c if c is not None else 'grey' for c in self.color_list.values()])
        plt.show()

############ CODE BLOCK 35 ################
    def color_the_graph(self):
        """
        This method colors the graph and returns if it was successful or not.

        :return: If the graph is colored
        :rtype: bool
        """
        nodes = list(self.adjacency_list.keys())
        return self._step(nodes)

    def _step(self, nodes):
        """
        One step in the coloring of the graph.

        :param nodes: A list of nodes that are not colored yet.
        :type nodes: list[int/str]
        """
        if not nodes:
            return True
        
        node = nodes[0]
        for color in self.colors:
            if self._is_correct(node, color):
                self.color_list[node] = color
                if self._step(self._next_step(nodes)):
                    return True
                self._clean_up(node)
        return False

    def _clean_up(self, node):
        """
        If all possible actions in step fail, it makes sure that any decisions are undone.

        :param node: The current node that is being colored
        :type node: int/str
        """
        self.color_list[node] = None
    
    def _next_step(self, nodes):
        """
        This method can help to go to the next step and it makes sure that the nodes list is correct.
        
        :param nodes: A list of nodes that are not colored yet.
        :type nodes: list[int/str]
        """
        return nodes[1:]
    
    def _is_correct(self, node):
        """
        This method checks if node is colored correctly.
        
        :param node: The current node that is being colored
        :type node: int/str
        """
        for neighbour in self.adjacency_list[node]:
            if self.color_list[neighbour] == self.color_list[node]:
                return False
        return True


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
