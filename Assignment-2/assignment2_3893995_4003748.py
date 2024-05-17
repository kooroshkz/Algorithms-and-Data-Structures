############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 1 ################

class FloodFillSolver():
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the coordinates that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    
    def __call__(self, road_grid, source, destination):
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both coordinates should be included in the path.
        To find the shortest route a version of a flood fill algorithm is used, see the explanation above.
        A route consists of a list of coordinates.

        Hint: The history is already given as a dictionary with as keys the coordinates in the state-space graph and
        as values the previous coordinate from which this coordinate was visited.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :param source: The coordinate where the path starts.
        :type source: tuple[int]
        :param destination: The coordinate where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of coordinates and the length of the route.
        :rtype: list[tuple[int]], float
        """
        self.queue = deque([source])
        self.history = {source: None}
        self.grid = road_grid
        self.destination = destination
        self.main_loop()
        return self.find_path()

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one coordinate to the next has a length of 1.
        For example: The distance between coordinates (0,0) and (0,1) is 1 and 
                     The distance between coordinates (3,0) and (3,3) is 3. 

        The distance is the Manhattan distance of the path.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        if self.destination not in self.history:
            return [], 0.0
        path = []
        current = self.destination
        while current:
            path.append(current)
            current = self.history.get(current)
        path.reverse()
        return path, float(len(path) - 1)
    
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.
        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue:
            current = self.queue.popleft()
            if self.base_case(current):
                return
            for new_node in self.next_step(current):
                self.step(current, new_node)

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the base case is found or not
        :rtype: bool
        """
        return node == self.destination
        
    def step(self, node, new_node):
        """
        One flood-fill step.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :param new_node: The next node/coordinate that can be visited from the current node/coordinate
        :type new_node: tuple[int]       
        """
        if new_node not in self.history:
            self.queue.append(new_node)
            self.history[new_node] = node

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        # possible problem
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible movements: right, down, left, up
        return [(node[0] + move[0], node[1] + move[1]) for move in moves
                if 0 <= node[0] + move[0] < self.grid.shape[0] and
                0 <= node[1] + move[1] < self.grid.shape[1] and
                self.grid[node[0] + move[0], node[1] + move[1]] != 0]

############ CODE BLOCK 10 ################

class GraphBluePrint():
    """
    You can ignore this class, it is just needed due to technicalities.
    """
    def find_nodes(self): pass
    def find_edges(self): pass
    
class Graph(GraphBluePrint):   
    """
    Attributes:
        :param adjacency_list: The adjacency list with the road distances and speed limit.
        :type adjacency_list: dict[tuple[int]: set[edge]], where an edge is a fictional datatype 
                              which is a tuple containing the datatypes tuple[int], int, float
        :param map: The map of the graph.
        :type map: Map
    """
    def __init__(self, map_, start=(0, 0)):
        """
        This function transforms any (city or lower) map into a graph representation.

        :param map_: The map that needs to be transformed.
        :type map_: Map
        :param start: The start node from which we will find all other nodes.
        :type start: tuple[int]
        """
        self.adjacency_list = {}
        self.map = map_
        self.start = start
        
        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell
        
    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        So far, we called this method `step`. However, this class is more than just the search algorithm,
        therefore, we gave it a bit more descriptive name.

        Note, that we only want to find the nodes, so history does not need to contain a partial path (previous node).
        In `find_edges` (the next cell), we will add edges for each node.
        """
        queue = deque([self.start])
        history = {self.start}
        self.adjacency_list[self.start] = set()
        while queue:
            current = queue.popleft()
            for new_node in self.neighbour_coordinates(current):
                if new_node not in history:
                    queue.append(new_node)
                    history.add(new_node)
                    self.adjacency_list_add_node(new_node, self.neighbour_coordinates(new_node))
        
                    
    def adjacency_list_add_node(self, coordinate, actions):
        """
        This is a helper function for the breadth-first search algorithm to add a coordinate to the `adjacency_list` and
        to determine if a coordinate needs to be added to the `adjacency_list`.

        Reminder: A coordinate should only be added to the adjacency list if it is a corner, a crossing, or a dead end.
                  Adding the coordinate to the adjacency_list is equivalent to saying that it is a node in the graph.

        :param coordinate: The coordinate that might need to be added to the adjacency_list.
        :type coordinate: tuple[int]
        :param actions: The actions possible from this coordinate, an action is defined as an action in the coordinate state-space.
        :type actions: list[tuple[int]]
        """
        # changed this method to account for corners, start and end
        width, height = len(self.map[0]) - 1, len(self.map[:,1]) - 1
        edge_cases = [(0, 0), (0,  width), (height, 0), (height, width)]
        
        if len(actions) == 2:
            action1, action2 = tuple(actions)
        else:
            action1, action2 = (0, 0), (0, 0)

        if len(actions) > 2 or len(actions) == 1 or (action1[0] != action2[0] and action1[1] != action2[1]):
            self.adjacency_list[coordinate] = set()
        

                           
    def neighbour_coordinates(self, coordinate):
        """
        This method returns the next possible actions and is part of the breadth-first search algorithm.
        Similar to `find_nodes`, we often call this method `next_step`.
        
        :param coordinate: The current coordinate
        :type coordinate: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [(coordinate[0] + move[0], coordinate[1] + move[1]) for move in moves
                if 0 <= coordinate[0] + move[0] < self.map.shape[0] and
                0 <= coordinate[1] + move[1] < self.map.shape[1] and
                self.map[coordinate[0] + move[0], coordinate[1] + move[1]] != 0]
    
    def __repr__(self):
        """
        This returns a representation of a graph.

        :return: A string representing the graph object.
        :rtype: str
        """
        # You can change this to anything you like, such that you can easily print a Graph object. An example is already given.
        return repr(dict(sorted(self.adjacency_list.items()))).replace("},", "},\n")

    def __getitem__(self, key):
        """
        A magic method that makes using keys possible.
        This makes it possible to use self[node] instead of self.adjacency_list[node]

        :return: The nodes that can be reached from the node `key`.
        :rtype: set[tuple[int]]
        """
        return self.adjacency_list[key]

    def __contains__(self, key):
        """
        This magic method makes it possible to check if a coordinate is in the graph.

        :return: This returns if the coordinate is in the graph.
        :rtype: bool
        """
        return key in self.adjacency_list

    def get_random_node(self):
        """
        This returns a random node from the graph.
        
        :return: A random node
        :rtype: tuple[int]
        """
        return tuple(RNG.choice(list(self.adjacency_list)))
        
    def show_coordinates(self, size=5, color='k'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.

        :param size: The size of the dots, default to 5
        :type size: int
        :param color: The Matplotlib color of the dots, defaults to black
        :type color: string
        """
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)        

    def show_edges(self, width=0.05, color='r'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.
        
        :param width: The width of the arrows, default to 0.05
        :type width: float
        :param color: The Matplotlib color of the arrows, defaults to red
        :type color: string
        """
        for node, edge_list in self.adjacency_list.items():
            for next_node,_,_ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1])*0.975, (next_node[0] - node[0])*0.975, color=color, length_includes_head=True, width=width, head_width=4*width)

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        # reworked this completely, should work now
        width, height = len(self.map[0]) - 1, len(self.map[:,1]) - 1
        for node in self.adjacency_list:
            node_edges = []
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                # check if we are not outside of the grid by going to the next step
                # by either going to negative values or values higher than the grid size
                # if (node[0] == 0 and direction[0] == -1) or (node[1] == 0 and direction[1] == -1):
                #     continue
                # elif (node[0] == height and direction[0] == 1) or (node[1] == width and direction[1] == 1):
                #     continue
                # elif self.map[node[0] + direction[0]][node[1] + direction[1]] == 0:
                #     continue
                
                new_height, new_width = node[0] + direction[0], node[1] + direction[1]
                if new_width < 0 or new_height < 0 or new_width > width or new_height > height:
                    continue
                if not self.map[new_height][new_width]:
                    continue
                next_node, distance = self.find_next_node_in_adjacency_list(node, direction)
                speed_limit = self.map[next_node[0], next_node[1]]
                node_edges.append((next_node, distance, speed_limit))
            self.adjacency_list[node] = node_edges


    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """
        current_node = node
        distance = 0
        while True:
            distance += 1
            new_node = (current_node[0] + direction[0], current_node[1] + direction[1])
            if new_node in self.adjacency_list:
                return new_node, distance
            current_node = new_node

############ CODE BLOCK 120 ################

class FloodFillSolverGraph(FloodFillSolver):
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the nodes that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __call__(self, graph, source, destination):      
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        Hint: The history is already given as a dictionary with as keys the node in the state-space graph and
        as values the previous node from which this node was visited.

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts.
        :type source: tuple[int]
        :param destination: The node where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of nodes and the length of the route.
        :rtype: list[tuple[int]], float
        """ 
        #IMPORTANT   
        # this code works if we consider the output distance to be the number of
        # nodes in a path, and not the distance in the tuples of the adjacency list
        # need to ask during the workgroup
        self.queue = deque([source])
        self.history = {source: None}
        self.graph = graph
        self.destination = destination
        self.main_loop()
        return self.find_path()

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one coordinate to the next has a length of 1.
        For example: The distance between coordinates (0,0) and (0,1) is 1 and 
                     The distance between coordinates (3,0) and (3,3) is 3. 

        The distance is the Manhattan distance of the path.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        if self.destination not in self.history:
            return [], 0.0
        path = []
        current = self.destination
        prev = self.destination
        length = 0.0
        while current:
            path.append(current)
            length += np.sqrt((current[0] - prev[0])**2 + (current[1] - prev[1])**2)
            prev = current
            current = self.history.get(current)
        path.reverse()
        return path, length

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return [item[0] for item in self.graph.adjacency_list[node]]

############ CODE BLOCK 130 ################

class BFSSolverShortestPath():
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the distances it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple(int), float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the distance it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], int]]
    """   
    def __call__(self, graph, source, destination):      
        """
        This method gives the shortest route through the graph from the source to the destination node.
        You start at the source node and the algorithm ends if you reach the destination node, 
        both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts
        :type source: tuple[int] 
        :param destination: The node where the path ends
        :type destination: tuple[int]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: The shortest route and the time it takes. The route consists of a list of nodes.
        :rtype: list[tuple[int]], float
        """       
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.source = source
        self.destination = destination
        self.graph = graph
        self.main_loop()
        return self.find_path()


    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        if self.destination not in self.history:
            return [], 0.0
        path = []
        current = self.destination
        prev = self.destination
        length = 0.0
        while current:
            path.append(current)
            next_item = self.history.get(current)
            # length += next_item[1]
            prev = current
            current = next_item[0]
        path.reverse()
        return path, self.history[self.destination][1]

    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.priorityqueue:
            self.priorityqueue = sorted(self.priorityqueue, key= lambda x: x[1])
            current, current_distance = self.priorityqueue.pop(0)
            if self.base_case(current):
                return
            for new_node in self.next_step(current):
                current_distance = np.sqrt((current[0] - new_node[0])**2 + (current[1] - new_node[1])**2)
                self.step(current, new_node, current_distance, graph.map.grid[new_node[0]][new_node[1]])

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        return node == self.destination

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        For now, speed_limit can be ignored.

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
   
        return distance + self.history[previous_node][1]

    def step(self, node, new_node, distance, speed_limit):
        """
        One step in the BFS algorithm. For now, speed_limit can be ignored.

        :param node: The current node
        :type node: tuple[int]
        :param new_node: The next node that can be visited from the current node
        :type new_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        """
        if new_node not in self.history:
            self.priorityqueue.append((new_node, 0))
            self.history[new_node] = (node, self.new_cost(node, distance, speed_limit))
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return [item[0] for item in self.graph.adjacency_list[node]]

############ CODE BLOCK 200 ################

class BFSSolverFastestPath(BFSSolverShortestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited 
                              including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the time it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):      
        """
        This method gives a fastest route through the grid from source to destination.

        This is the same as the `__call__` method from `BFSSolverShortestPath` except that 
        we need to store the vehicle speed. 
        
        Here, you can see how we can overwrite the `__call__` method but 
        still use the `__call__` method of BFSSolverShortestPath using `super`.
        """
        self.vehicle_speed = vehicle_speed
        return super(BFSSolverFastestPath, self).__call__(graph, source, destination)

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        Use the `speed_limit` and `vehicle_speed` to determine the time/cost it takes to go to
        the new node from the previous_node and add the time it took to reach the previous_node to it..

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        min_speed = min(speed_limit, self.vehicle_speed)
        time_to_new_node = distance / min_speed
        total_time = time_to_new_node + self.history[previous_node][1]

        return total_time

############ CODE BLOCK 210 ################

def coordinate_to_node(map_, graph, coordinate):
    """
    This function finds a path from a coordinate to its closest nodes.
    A closest node is defined as the first node you encounter if you go a certain direction.
    This means that unless the coordinate is a node, you will need to find two closest nodes.
    If the coordinate is a node then return a list with only the coordinate itself.
    :param map_: The map of the graph
    :type map_: Map
    :param graph: A Graph of the map
    :type graph: Graph
    :param coordinate: The coordinate from which we want to find the closest node in the graph
    :type coordinate: tuple[int]
    :return: This returns a list of closest nodes which contains either 1 or 2 nodes.
    :rtype: list[tuple[int]]
    """
    closest_nodes = []
    if coordinate in graph.adjacency_list:
        return [coordinate]

    width, height = len(graph.map[0]) - 1, len(graph.map[:,1]) - 1 
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_height, new_width = coordinate[0] + direction[0], coordinate[1] + direction[1]
        if new_width < 0 or new_height < 0 or new_width > width or new_height > height:
            continue
        if not graph.map[new_height][new_width]:
            continue
        closest_nodes.append(find_node_in_direction(graph, (new_height, new_width), direction))
    return closest_nodes

# added this helper function
def find_node_in_direction(graph, node, direction):
    """
    This function finds the closest node in a certain direction.

    :param graph: The graph of the map
    :type graph: Graph
    :param node: The node from which we want to find the closest node in the graph
    :type node: tuple[int]
    :param direction: The direction in which we want to find the closest node
    :type direction: tuple[int]
    """
    current = node
    while True:
        if current in graph.adjacency_list:
            return current
        current = (current[0] + direction[0], current[1] + direction[1])

############ CODE BLOCK 220 ################

def create_country_graphs(map_):
    """
    This function returns a list of all graphs of a country map, where the first graph is the highways and de rest are the cities.

    :param map_: The country map
    :type map_: Map
    :return: A list of graphs
    :rtype: list[Graph]
    """
    graphs = []
    highway = map_.get_highway_map()
    start_highway = map_.get_all_city_exits()[0]
    highway_graph = Graph(highway, start_highway)
    graphs.append(highway_graph)
    
    city_map = map_.get_city_map()
    # city corners is a list with the starting points of every city
    # its length is equal to the number of cities
    city_starts = map_.city_corners
    for city_index in range(len(city_starts)):
        city_start = city_starts[city_index]
        city_graph = Graph(city_map, city_start)
        graphs.append(city_graph)

    return graphs

############ CODE BLOCK 230 ################

class BFSSolverMultipleFastestPaths(BFSSolverFastestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that are visited and as values the node that leads to this node including the time it takes from the start node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
        :param found_destinations: The destinations already found with Dijkstra.
        :type found_destinations: list[tuple[int]]
    """
    def __init__(self, find_at_most=3):
        """
        This init makes it possible to make a different Dijkstra algorithm 
        that find more or less destination nodes before it stops searching.

        :param find_at_most: The number of found destination nodes before the algorithm stops
        :type find_at_most: int
        """
        self.find_at_most = find_at_most
    
    def __call__(self, graph, sources, destinations, vehicle_speed):      
        """
        This method gives the top three fastest routes through the grid from any of the sources to any of the destinations.
        You start at the sources and the algorithm ends if you reach enough destinations, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param sources: The nodes where the path starts and the time it took to get here.
        :type sources: list[tuple[tuple[int], float]]
        :param destinations: The nodes where the path ends and the time it took to get here.
        :type destinations: list[tuple[tuple[int], float]]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """       
        self.priorityqueue = sorted(sources, key=lambda x:x[1])
        self.history = {s: (None, t) for s, t in sources}
        
        if self.find_at_most > 0 and self.find_at_most < len(destinations):
            self.destinations = destinations[:self.find_at_most]
        else:
            self.destinations = destinations
        self.destination_nodes = [dest[0] for dest in destinations]
        self.found_destinations = []
        self.graph = graph
        self.sources = sources
        self.vehicle_speed = vehicle_speed
        self.main_loop()
        return self.find_n_paths()


    def find_n_paths(self):
        """
        This method needs to find the top `n` fastest paths between any source node and any destination node.
        This does not mean that each source node has to be in a path nor that each destination node needs to be in a path.

        Hint1: The fastest path is stored in each node by linking to the previous node. 
               Therefore, if you start searching from a destination node,
               you always find the optimal path from that destination node.
               This is similar if you only had one destination node.         

        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """
        # return [(start[0], end) for end, start in self.history.items() if start[0] is not None]
        paths = []
        for destination in self.found_destinations:
            current = destination
            path = []
            time = self.history[current][1]
            while current:
                path.append(current)
                new = self.history[current][0]
                current = new
            path.reverse()
            paths.append((path, time))
        
        return sorted(paths, key=lambda x: x[1])
            
        
    def base_case(self, node):
        """
        This method checks if the base case is reached and
        updates self.found_destinations

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        if node in self.destination_nodes:
            self.found_destinations.append(node)
            
        if len(self.found_destinations) == len(self.destination_nodes):
            return True
        return False

############ CODE BLOCK 235 ################

class BFSSolverFastestPathMD(BFSSolverFastestPath):
    def __call__(self, graph, source, destinations, vehicle_speed):      
        """
        This method is functionally no different than the call method of BFSSolverFastestPath
        except for what `destination` is.

        See for an explanation of all arguments `BFSSolverFastestPath`.
        
        :param destinations: The nodes where the path ends.
        :type destinations: list[tuple[int]]
        """
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.destinations = destinations
        self.destination = None
        self.vehicle_speed = vehicle_speed

        raise NotImplementedError("Please complete this method")       

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: returns True if the base case is reached.
        :rtype: bool
        """
        raise NotImplementedError("Please complete this method")

############ CODE BLOCK 300 ################

def path_length(coordinate, closest_nodes, map_, vehicle_speed):
    return [(node, (abs(node[0] - coordinate[0]) + abs(node[1] - coordinate[1])) / min(vehicle_speed, map_[coordinate])) for node in closest_nodes] 

def find_path(coordinate_A, coordinate_B, map_, vehicle_speed, find_at_most=3):
    """
    Find the optimal path according to the divide and conquer strategy from coordinate A to coordinate B.

    See hints and rules above on how to do this.

    :param coordinate_A: The start coordinate
    :type coordinate_A: tuple[int]
    :param coordinate_B: The end coordinate
    :type coordinate_B: tuple[int]
    :param map_: The map on which the path needs to be found
    :type map_: Map
    :param vehicle_speed: The maximum vehicle speed
    :type vehicle_speed: float
    :param find_at_most: The number of routes to find for each path finding algorithm, defaults to 3. 
                         Note, that this is only needed if you did 2.3.
    :type find_at_most: int, optional
    :return: The path between coordinate_A and coordinate_B. Also, return the cost.
    :rtype: list[tuple[int]], float
    """
    graphs = create_country_graphs(map_)
    highway_graph = graphs[0]
    country = Graph(map_)
    city_graphs = graphs[1:]
    city_map = map_.get_city_map()
   
    BFSMP = BFSSolverMultipleFastestPaths(find_at_most)
    nodes_A = coordinate_to_node(map_, country, coordinate_A)
    nodes_B = coordinate_to_node(map_, country, coordinate_B)
    nodes_time_A = path_length(coordinate_A, nodes_A, map_, vehicle_speed)
    nodes_time_B = path_length(coordinate_B, nodes_B, map_, vehicle_speed)

    city_A, city_B = find_city(nodes_A[0], city_graphs), find_city(nodes_B[0], city_graphs) 
    all_city_exits = map_.get_all_city_exits()

    # find exits for city B and city B and change the datatype for city exits 
    # to make them compatible with the BFS solver
    city_exits_A, city_exits_B = [(node, 0) for node in all_city_exits if node in city_graphs[city_A]], [(node, 0) for node in all_city_exits if node in city_graphs[city_B]]
    
    # find the shortest time, path from any node A to any exit A
    path_A, cost_A = min(BFSMP(city_graphs[city_A], nodes_time_A, city_exits_A, vehicle_speed), key=lambda x:x[1])
    A_coordinate_to_exit = [coordinate_A] + list(path_A)

    # find the shortest time, path from any node B to any exit B
    path_B, cost_B = min(BFSMP(city_graphs[city_B], city_exits_B, nodes_time_B, vehicle_speed), key=lambda x:x[1])
    B_exit_to_coordinate = list(path_B) + [coordinate_B]
    
    # find the path and time through the highway
    exit_A, exit_B = path_A[-1], path_B[0]
    middle_path, middle_cost = BFSSolverFastestPath()(highway_graph, exit_A, exit_B, vehicle_speed)

    total_path, total_cost = A_coordinate_to_exit + middle_path + B_exit_to_coordinate, cost_A + middle_cost + cost_B
    # remove duplicate nodes for example when we add different paths or a coordinate is a node in a graph
    total_path = list(dict.fromkeys(total_path))
    
    # check if we are in the same city
    # then we have to compare going through the city or going through the highway
    if city_A == city_B:
        path_in_city, cost_in_city = min(BFSMP(city_graphs[city_A], nodes_time_A, nodes_time_B, vehicle_speed), key=lambda x:x[1])
        path_in_city_coordinates = [coordinate_A] + list(path_in_city) + [coordinate_B]
        path_in_city_coordinates = list(dict.fromkeys(path_in_city_coordinates))
        candidates = [(total_path, total_cost), (path_in_city_coordinates, cost_in_city)]
        return min(candidates, key=lambda x:x[1])
        
    return total_path, total_cost

# added this helper function 
def find_city(node, city_graphs):
    """
    Find the index of the city in a list of graphs,
    that contains the given node

    :param node: The node that is in a city
    :type node: tuple[int]
    :param city_graphs: A list of graphs where each graph is a city
    :type city_graphs: list[Graph]
    """
    for index, city in enumerate(city_graphs):
        if node in city:
            return index


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
