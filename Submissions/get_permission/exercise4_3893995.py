############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
def alternating_disks(n):
    """
    This function solves the alternating disks problem for a row with size 2*n containing n light disks and n dark disks.
    The function returns an array with n ones (light disks) and n zeros (dark disks).
    Use the function swap to change the position of two disks.
    
    :param n: Number of light or dark disks
    :type n: int
    :return: The ordered list (light is 1 and dark is 0)
    :rtpye: np.ndarray[(2*n), int]
    """
    disks = np.array(list(zip(np.ones(n), np.zeros(n)))).flatten()
    return disks

def swap(disks, i):
    """
    This function swaps the disks i and i+1 in the array disks.
    This is a helper function for alternating_disks.

    :param disks: Array containing the light and dark disks
    :type disks: np.ndarray[(2*n), int]
    :param i: Position of the disk that needs to be swapped
    :type i: int
    """
    pass

############ CODE BLOCK 20 ################
class Permutations():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating them first using step and
        then filtering them using `self.is_correct`.

        :param list_: The list containing all the elements.
        :type list_: list[Objects]
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # Nothing needs to be initialized or filter. So just call self.step
        return self.step(list_)
        
    def step(self, list_):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.

        :param list_: The list containing all the possible elements (that are valid actions)
        :type list_: list
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        if len(list_) == 0:
            return [[]]

        all_permutations = []
        for i, elem in enumerate(list_):
            next_permutations = self.next_step(list_[:i] + list_[i+1:], i)
            for perm in next_permutations:
                all_permutations.append([elem] + perm)
        return all_permutations

    def next_step(self, list_, index):
        """
        This method generates the actions that are possible for the next step and calls step.
        These actions consist of the elements of the list that are not yet in the new permutated list.

        :param list_: The list containing all the possible elements (that are valid actions)
                      from the previous state-space.
        :type list_: list
        :param index: The index of the element that is used as action in the previous step.
        :type index: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        return self.step(list_)

############ CODE BLOCK 25 ################
class PermutationsWithReplacement():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating them first using step and
        then filtering them using `self.is_correct`.
        
        :param list_: The list containing all the unique elements.
        :type list_: list
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # all actions are the same, so it is helpful to make an object attribute.
        self.list = list_
        # filter out all incorrect state-spaces and return all permutations of self.list
        return [x for x in self.step(0) if self.is_correct(x)]

    def step(self, i):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.
        
        :param i: A counter how many elements are added to the new permutation.
        :type i: int
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        if i == len(self.list):
            return [[]]

        all_permutations = []
        for elem in self.list:
            next_permutations = self.next_step(i)
            for perm in next_permutations:
                all_permutations.append([elem] + perm)
        return all_permutations

    def next_step(self, i):
        """
        This method generates the actions that are possible for the next step and calls step.
        These actions consist of all elements of the original list.

        :param i: An counter how many elements are added to the new permutation.
        :type i: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        return self.step(i + 1)
    
    def is_correct(self, permutation):
        """
        This method returns if the state-space is correct aka if it is a permutation.

        :param permutation: A possible permutation of self.list
        :type permutation: list[Objects]
        :return: Return if the permutation variable is or is not a permutation.
        :rtype: boolean
        """
        return len(set(permutation)) == len(self.list)

############ CODE BLOCK 30 ################
class ManhattanProblem():
    def __call__(self, road_grid):
        """
        This method gives all the fastest routes through this part of Manhattan.
        You always start top left and end bottom right.
        This is done by calling step which should return a list of routes, 
        where a route consists of a list of coordinates.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :return: A list with all possible routes, where a route consists of a list of coordinates.
        :rtype: list[list[tuple[int]]]
        """
        self.grid = road_grid
        return self.next_step((0,0))  # We already are in the first state-space so we need to generate the next actions.
        
    def step(self, pos, actions):
        """
        This method does one step in the depth-first search in the Manhattan grid.
        One step consists of adding one coordinate (tuple) to the route and
        generating all possible routes from this coordinate in the grid,
        this is done recursively.

        :param pos: The current coordinate in the grid.
        :type pos: tuple[int]
        :param actions: List of possible next coordinates.
        :type actions: list[tuple[int]]
        :return: All possible route with this position as starting point.
        :rtype: list[list[tuple[int]]]
        """
        routes = []
        for action in actions:
            new_pos = (pos[0] + action[0], pos[1] + action[1])
            if 0 <= new_pos[0] < len(self.grid) and 0 <= new_pos[1] < len(self.grid[0]):
                routes.extend(self.next_step(new_pos))
        return routes

    def next_step(self, pos):
        """
        Here, we check which actions we can take depending on the current position in the grid.
        Then, we call next step with the current position and next possible actions.

        :param pos: The current coordinate in the grid.
        :type pos: tuple[int]
        :return: This method returns what self.step returns
        :rtype: list[list[tuple[int]]]
        """
        actions = [(1, 0), (0, 1)]  # Right, Down
        if pos == (len(self.grid) - 1, len(self.grid[0]) - 1):  # Reached bottom right corner
            return [[pos]]
        return self.step(pos, actions)

############ CODE BLOCK 32 ################
class Node():
    """
    This Node class forms a graph where each node contains directed edges to the other nodes.

    Attributes:
        :param info: The coordinate of the Node
        :type info: tuple[(2,), int]
        :param edges: A list of Nodes which are the directed edges from this Node.
        :type edges: list[Node]
    """
    def __init__(self, info):
        self.info = info
        self.edges = []

    def set_edges(self, edges):
        self.edges = edges

    def __repr__(self):
        return f"Node{self.info}"  # you can choose which representation you like or make one yourself.
        return f"Node{self.info} -> {[e.info for e in self.edges]}"

############ CODE BLOCK 33 ################
def make_graph(grid):
    # if the grid is effectively one dimensional
    if 1 in grid.shape:
        start = Node((0,0))
        start.edges = [Node((grid.shape[0]-1, grid.shape[1]-1))]
        return start
    
    # add start and end nodes
    row_col_nodes = defaultdict(list)
    col_row_nodes = defaultdict(list)
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            row_adj = sum([row[j-1], row[j+1]] if j > 0 and j < grid.shape[1]-1 else [row[j+1] if j == 0 else row[j-1]])  # check if there is an adjacent road horizontally
            col_adj = sum([grid[i-1, j], grid[i+1, j]] if i > 0 and i < grid.shape[0]-1 else [grid[i+1,j] if i == 0 else grid[i-1,j]]) # check if there is an adjacent road vertically
            # junction if both vertically and horizontally there is a road and you are standing on a road. 
            if cell and row_adj and col_adj:
                row_col_nodes[i] += [j]
                col_row_nodes[j] += [i]

    # make nodes
    nodes = {}
    for row, cols in row_col_nodes.items():
        for col in cols:
            nodes[(row, col)] = Node((row,col))

    # connect nodes
    for (row, col), node in nodes.items():
        edges = []
        if col + 1 < grid.shape[1] and grid[row, col + 1]:  # go left and find next node
            edges.append(nodes[row, row_col_nodes[row][row_col_nodes[row].index(col)+1]])
        if row + 1 < grid.shape[0] and grid[row + 1, col]: # go down and find next node
            edges.append(nodes[col_row_nodes[col][col_row_nodes[col].index(row)+1], col])
        node.set_edges(edges)
            
    return nodes[(0,0)]

############ CODE BLOCK 35 ################
class ManhattanProblemDepth():
    def __call__(self, road_graph):
        """
        This method gives all the fastest routes through this part of Manhattan.
        You start with the first node `road_graph` and you end if you are at the end node.
        You can assume there are no dead ends in the graph.
        This is done by calling step which should return a list of routes, 
        where a route consists of a list of coordinates.

        :param road_graph: The start Node of the graph.
        :type road_graph: Node
        :return: A list with all possible routes, where a route consists of a list of coordinates.
        :rtype: list[list[tuple[int]]]
        """
        return self.step(road_graph)
        
    def step(self, node):
        """
        This method does one step in the depth-first search in the Manhattan grid.
        One step consists of adding one coordinate (tuple) to the route and
        generating all possible routes from this coordinate in the grid.

        :param node: A Node, that contains the current coordinate in the grid.
        :type node: Node
        :return: All possible routes with this position as starting point.
        :rtype: list[list[tuple[int]]]
        """
        routes = []
        for edge in node.edges:
            routes.extend(self.next_step(edge))
        return routes

    def next_step(self, node):
        """
        Becaues, we are traversing the state-space graph itself explicitly, 
        there is nothing to do in next_step, as the next actions are encoded by the edges.

        :param node: A Node, that contains the current coordinate in the grid.
        :type node: Node
        :return: This method returns what self.step returns
        :rtype: list[list[tuple[int]]]
        """
        return self.step(node)

############ CODE BLOCK 37 ################
class ManhattanProblemBreadth():   
    def __call__(self, road_graph):      
        """
        This method gives all the fastest routes through this part of Manhattan.
        You start with the first node `road_graph` and you end if you are at the end node.
        You can assume there are no dead ends in the graph.

        Hint: The history is already given as a dictionary with as keys all the nodes in the state-space graph and
        as values all possible routes that lead to this node.

        This class instance should at least contain the following attributes after being called:
            :param queue: A queue that contains all the nodes that need to be visited.
            :type queue: collections.deque
            :param history: A dictionary containing the nodes that are visited and all routes that lead to that node.
            :type history: dict[Node, set[tuple[tuple]]]

        :param road_graph: The start Node of the graph.
        :type road_graph: Node
        :return: A list with all possible routes, where a route consists of a list of coordinates.
        :rtype: list[list[tuple[int]]]
        """
        self.queue = deque([road_graph])  # propper datastructure, however, a list would work as well
        self.history = {road_graph: {(road_graph.info,)}}
        
        self.main_loop()
        return [list(route) for route in self.history[road_graph]]

    def main_loop(self):
        """
        This method contains the logic of the breadth-first search for the Manhattan problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue:
            node = self.queue.popleft()
            if self.base_case(node):
                continue
            for new_node in self.next_step(node):
                self.step(node, new_node)

    def base_case(self, node):
        """
        This method checks if the current node is the base code, i.e., final node.

        :param node: The current node
        :type node: Node
        """
        if node.info == (len(road_grid) - 1, len(road_grid[0]) - 1):
            return True
        return False

    def step(self, node, new_node):
        """
        One breadth-first search step.
        Here, you add new nodes to the queue and update the history.

        :param node: The current node
        :type node: Node
        :param new_node: The next node that can be visited from the current node
        :type new_node: Node        
        """
        if new_node not in self.history:
            self.queue.append(new_node)
            self.update_history(node, new_node)
        
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: Node
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[Node]  
        """
        return [edge.destination for edge in node.edges]
        
    def update_history(self, node, new_node):
        """
        For more complex histories it is good to have a separate method to 
        set or update the history.
        
        :param node: The current node
        :type node: Node
        :param new_node: The next node that can be visited from the current node
        :type new_node: Node    
        """
        new_routes = set()
        for route in self.history[node]:
            new_routes.add(route + (new_node.info,))
        self.history[new_node] = new_routes


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
