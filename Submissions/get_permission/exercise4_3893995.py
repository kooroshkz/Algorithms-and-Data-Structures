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
    i = 0
    while 1 in disks[n:] and i < 2 * n - 2:
        if disks[i] != disks[i + 1] and disks[i] == 0:
            swap(disks, i)
        i += 1
        if i == 2 * n - 2:
            i = 0
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
    disks[i], disks[i + 1] = disks[i + 1], disks[i]

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
        
        
        permutations = []
        for index in range(len(self.list)):
            # call the next step to generate available actions
            for action in self.next_step(i):
                action.append(self.list[index])
                permutations.append(action)
                
        return permutations

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
        if pos == (len(self.grid) - 1, len(self.grid[0]) - 1):
            return [[pos]]
        
        paths = []
        for action in actions:
            for path in self.next_step(action):
                paths.append([pos] + path)
        
        return paths

    def next_step(self, pos):
        """
        Here, we check which actions we can take depending on the current position in the grid.
        Then, we call next step with the current position and next possible actions.

        :param pos: The current coordinate in the grid.
        :type pos: tuple[int]
        :return: This method returns what self.step returns
        :rtype: list[list[tuple[int]]]
        """
        pos_candidates = []
        grid = self.grid
        if pos[0] + 1 < len(grid) and grid[pos[0] + 1][pos[1]] == 1:
            pos_candidates.append((pos[0] + 1, pos[1]))
        if pos[1] + 1 < len(grid[0]) and grid[pos[0]][pos[1] + 1] == 1:
            pos_candidates.append((pos[0], pos[1] + 1))
        
        return self.step(pos, pos_candidates)

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
        self.queue = deque([road_graph])
        self.history = {road_graph: {(road_graph.info,)}}
        self.final_node = None
        self.main_loop()
        output_list = sorted(list(self.history[self.final_node]))
        new_list = [list(path) for path in output_list]
        return new_list

    def main_loop(self):
        """
        This method contains the logic of the breadth-first search for the Manhattan problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue:
            current_node = self.queue.popleft()
            if self.base_case(current_node):
                 continue
            for sub_node in self.next_step(current_node):
                self.step(current_node, sub_node)

    def base_case(self, node):
        """
        This method checks if the current node is the base code, i.e., final node.

        :param node: The current node
        :type node: Node
        """
        if not node.edges:
            self.final_node = node
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
        if node not in self.queue:
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
        return node.edges
        
    def update_history(self, node, new_node):
        """
        For more complex histories it is good to have a separate method to 
        set or update the history.
        
        :param node: The current node
        :type node: Node
        :param new_node: The next node that can be visited from the current node
        :type new_node: Node    
        """
        if new_node not in self.history.keys():
            self.history[new_node] = set()
        for path in self.history[node]: 
            self.history[new_node].add(path + (new_node.info,))

############ CODE BLOCK 40 ################
class TowerOfHanoiFast():
    def __call__(self, n):
        """
        This method solves the tower of Hanoi according to the fast algorithm.
        The output should be a list containing all the moves to solve the problem.
        A move is a tuple with two integers containing from which rod to which rod.
        For example, move (0,2) is place the top disk of rod 0 onto rod 2.
        Also, the attribute towers should contain the end solution where all disks are 
        in the correct order on rod 2.

        This class instance should at least contain the following attributes after being called:
            :param tower: A tuple with the tree rods each rod is represented by a list 
                          with integers where the integer is the size of the disk.
            :type tower: tuple[list[int]]

        Hint 1: It might be easier to first generate the moves and then apply them to the towers.
        Hint 2: The argument "n" is not needed for the algorithm but can help. 
                It is however needed if you do hint 1.

        :param n: This is the height of the tower, i.e., the number of disks.
        :type n: int
        :return: A list containing all the moves to solve the tower of Hanoi with n disks.
        :rtype: list[tuple[int]]
        """
        self.towers = (list(range(n, 0, -1)), [], [])

        self.towers = (list(range(n, 0, -1)), [], [])
        self.moves = []

        def move_disk(source, dest):
            disk = self.towers[source].pop()
            self.towers[dest].append(disk)
            self.moves.append((source, dest))

        def hanoi(n, source, aux, dest):
            if n == 1:
                move_disk(source, dest)
            else:
                hanoi(n - 1, source, dest, aux)
                move_disk(source, dest)
                hanoi(n - 1, aux, source, dest)

        hanoi(n, 0, 1, 2)
        return self.moves

    def step(self, source, aux, dest, n=0):
        """
        This method can be used as the recursive part of the algorithm

        :param source: The rod that has the current (sub)tower that needs to be moved.
        :type source: int
        :param aux: The auxiliary rod that can be used to transfer all disks to the destination rod.
        :type aux: int
        :param dest: The destination rod where the (sub)tower needs to go.
        :type dest: int
        :param n: The height of the current tower that is moved, default to 0.
        :type n: int, optional
        """
        pass

############ CODE BLOCK 44 ################
class TowerOfHanoiDepth():
    # All possible actions for any tower of Hanoi State-space
    possible_actions = [(0, 1), 
                        (0, 2), 
                        (1, 2), 
                        (2, 1), 
                        (1, 0), 
                        (2, 0)] 

    def __call__(self, n):
        """
        This method uses depth-first search to find a solution to solve the tower of Hanoi.
        The output should be a list containing all the moves to solve the problem.
        A move is a tuple with two integers containing from which rod to which rod.
        For example, move (0,2) is place the top disk of rod 0 onto rod 2.
        Also, the attribute towers should contain the end solution where all disks are 
        in the correct order on rod 2.

        This class instance should at least contain the following attributes after being called:
            :param tower: A tuple with the tree rods each rod is represented by a list 
                          with integers where the integer is the size of the disk.
            :type tower: tuple[list[int]]
            :param moves: A list with all the moves to solve the problem
            :typem moves: list[tuple[int]]
            :param history: A set containing all states that are already visited.
            :type history: set[tuple[tuple[int]]]

        :param n: This is the height of the tower, i.e., the number of disks.
        :type n: int
        :return: A list containing all the moves to solve the tower of Hanoi with n disks.
        :rtype: list[tuple[int]]
        """
        self.moves = []  # a list to store the moves
        self.towers = (list(range(n, 0, -1)), [], [])
        self.history = {self.to_hashable_state(self.towers)}
        self.next_step()  # we first need the know which the next actions can be before we can take a step
        return self.moves

    @staticmethod
    def to_hashable_state(state):
        """
        This method makes a state into a hashable object.
        
        As we have seen last week sets can quickly find objects.
        However, to do this these objects must be hashable.
        A simple rule to know if a type is hashable is to ask if it is immutable,
        if not then often it is also not hashable because if you change the value
        the output of the hash function would also change.

        :param state: A current state of the tower of Hanoi problem, i.e., the current rod disk configuration.
        :type state: tuple[list[int]]
        :return: A state space that is hashable. In this case, also immutable.
        :rtype: tuple[tuple[int]]
        """
        return tuple(tuple(rod) for rod in state)

    def step(self, actions):
        """
        One step in the recursive depth-first search algorithm.

        Hint1: Do not forget to check if state has already been visited and 
               update the history as needed.
        Hint2: You only need to find one path, so as long as the path is correct
               you do not to explore any more possible actions.

        :param actions: A set of correct actions that can taken from this state.
        :type actions: list[tuple[int]]
        :return: If the current step is correct or not
        :rtype: boolean
        """
        if self.towers == ([], [], list(range(len(self.towers[0]), 0, -1))):
            return True
        for action in actions:
            if self.do_move(action):
                if self.to_hashable_state(self.towers) not in self.history:
                    self.history.add(self.to_hashable_state(self.towers))
                    self.moves.append(action)
                    if self.step(self.next_step()):
                        return True
                    self.clean_up(action)
                    self.moves.pop()
        return False
    
    def next_step(self):
        """
        This method helps us determine the next set of correct actions.
        This set of correct actions should be a subset of the class attribute `possible_actions`.

        :return: If the current step is correct or not
        :rtype: boolean
        """
        actions = []
        for move in self.possible_actions:
            if self.towers[move[0]] and (not self.towers[move[1]] or self.towers[move[0]][-1] < self.towers[move[1]][-1]):
                actions.append(move)
        return actions

    def do_move(self, action):
        """
        This is a helper method that does one move.
        One move consists of changing one disk from one rod to another and
        to save it the move.
        
        :param action: A correct action that is taken from this state.
        :type action: tuple[int]
        """
        if self.towers[action[0]]:
            disk = self.towers[action[0]].pop()
            self.towers[action[1]].append(disk)
            return True
        return False

    def clean_up(self, action):
        """
        Clean up the previous move, if the current action taken does not lead to a correct solution.
        For example, you got stuck because there are no moves that go to a state that is not visited.

        :param action: A correct action that is taken from this state.
        :type action: tuple[int]
        """
        if self.towers[action[1]]:
            disk = self.towers[action[1]].pop()
            self.towers[action[0]].append(disk)

############ CODE BLOCK 48 ################
class TowerOfHanoiBreadth():   
    # All possible actions for any tower of Hanoi State-space
    possible_actions = [(0, 1), 
                        (0, 2), 
                        (1, 2), 
                        (2, 1), 
                        (1, 0), 
                        (2, 0)] 

    def __call__(self, n):      
        """
        This method uses breadth-first search to find a solution to solve the tower of Hanoi.
        The output should be a list containing all the moves to solve the problem.
        A move is a tuple with two integers containing from which rod to which rod.
        For example, move (0,2) is place the top disk of rod 0 onto rod 2.
        Also, the attribute towers should contain the end solution where all disks are 
        in the correct order on rod 2.

        This class instance should at least contain the following attributes after being called:
            :param moves: A list with all the moves to solve the problem
            :typem moves: list[tuple[int]]
            :param history: A dictionary containing all states that are already visited as keys 
                            and with the values the moves to get there.
            :type history: dict[tuple[tuple[int]], list[tuple[int]]]

        :param n: This is the height of the tower, i.e., the number of disks.
        :type n: int
        :return: A list containing all the moves to solve the tower of Hanoi with n disks.
        :rtype: list[tuple[int]]
        """
        towers = (list(range(n, 0, -1)), [], [])
        self.queue = deque([copy.deepcopy(towers)])  # proper datastructure, however, a list would work as well
        self.history = {self.to_hashable_state(towers): []}
        
        self.main_loop()
        return self.moves

    def main_loop(self):
        """
        This method contains the logic of the breadth-first search for the towers of Hanoi problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue:
            towers = self.queue.popleft()
            if self.base_case(towers):
                continue
            for action in self.next_step(towers):
                self.step(towers, action)

    @staticmethod
    def to_hashable_state(towers):
        """
        This method makes a state into a hashable object.
        
        As we have seen last week sets can quickly find objects.
        However, to do this these objects must be hashable.
        A simple rule to know if a type is hashable is to ask if it is immutable,
        if not then often it is also not hashable because if you change the value
        the output of the hash function would also change.

        :param towers: A current state of the tower of Hanoi problem, i.e., the current rod disk configuration.
        :type towers: tuple[list[int]]
        :return: A state space that is hashable. In this case, also immutable.
        :rtype: tuple[tuple[int]]
        """
        return tuple(tuple(rod) for rod in towers)
    
    def base_case(self, towers):
        """
        This method checks if the current state is the final state, where
        all disks are on the last rod in the correct order.

        :param tower: A tuple with the tree rods each rod is represented by a list 
                      with integers where the integer is the size of the disk.
        :type tower: tuple[list[int]]
        """
        if towers[2] == list(range(len(towers[2]), 0, -1)):
            self.moves = self.history[self.to_hashable_state(towers)]
            self.towers = towers
            return True
        return False

    def step(self, towers, action):
        """
        One breadth-first search step.
        Here, you add new states to the queue, and update the history.

        Hint: To create a new state, you need to make a copy of the current towers and
              then adjust them otherwise the towers for all states are adjusted.
        
        :param tower: A tuple with the tree rods each rod is represented by a list 
                      with integers where the integer is the size of the disk.
        :type tower: tuple[list[int]]
        :param action: A correct action that is taken from this state.
        :type action: tuple[int]
        """
        source, dest = action
        disk = towers[source].pop()
        towers[dest].append(disk)
        self.queue.append(copy.deepcopy(towers))
        self.history[self.to_hashable_state(towers)] = self.history[self.to_hashable_state(towers)] + [action]

    def next_step(self, towers):
        """
        This method helps us determine the next set of correct actions.
        This set of correct actions should be a subset of the class attribute `possible_actions`.
        
        :param tower: A tuple with the tree rods each rod is represented by a list 
                      with integers where the integer is the size of the disk.
        :type tower: tuple[list[int]]
        :return: The list of possible next actions
        :rtype: list[tuple[int]
        """
        actions = []
        for action in self.possible_actions:
            source, dest = action
            if not towers[source]:
                continue
            if not towers[dest] or towers[dest][-1] > towers[source][-1]:
                actions.append(action)
        return actions


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
