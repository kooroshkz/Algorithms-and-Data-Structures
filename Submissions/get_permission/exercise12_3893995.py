############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################

class Graph():
    """
    An undirected graph (not permitting loops) class with the ability to color edges.
    
    Object attributes:
        :param adjacency_list: The representation of the graph.
        :type adjacency_list: dict[int, dict[int, int]]
    """    
    def __init__(self):
        """
        The graph is always initialized empty, use the `set_graph` method to fill it.
        """
        self.adjacency_list = {}
    
    def generate_random_graph(self, min_=6, max_=10):
        """
        This is a helper method to generate a random connected graph with random weights

        :param min_: The minimum size of the graph.
        :type min_: int
        :param max_: The maximum size of the graph.
        :type max_: int
        """
        size = RNG.integers(min_, max_)
        
        # add nodes
        self.adjacency_list = {i: {} for i in range(size)}

        # add edgesf
        node = int(RNG.integers(size))
        connected = {node}
        while len(connected) < size:
            # create between 1 and 5 edges for each node
            for edge in RNG.integers(size, size=RNG.integers(2, min(size, 5))):
                # skip self looping edges
                if edge == node:  
                    continue
                    
                self.adjacency_list[node][edge] = self.adjacency_list[edge][node] = RNG.integers(1, 10)  # undirected
                connected.add(edge)

            # create edges from a new node (which is the last edge from the current node)
            node = edge
            
    def set_graph(self, adjacency_list):
        """
        This method sets the graph using as input an adjacency list.

        :param adjacency_list: The representation of the weighted graph.
        :type adjacency_list: dict[int, dict[int, int]]
        """
        self.adjacency_list = adjacency_list

    def __getitem__(self, key):
        """
        A magic method that makes using keys possible.
        This makes it possible to use self[node] instead of self.adjacency_list[node], where node is an int.

        :return: The nodes that can be reached from the node `key` a.k.a the edges.
        :rtype: dict[int, int]
        """
        return self.adjacency_list[key]

    def get_random_node(self):
        """
        This returns a random node from the graph.
        
        :return: A random node
        :rtype: int
        """
        return RNG.choice(list(self.adjacency_list))

    def __repr__(self):
        """
        The representation of the graph
        """
        return repr(self.adjacency_list)
    
    def show(self, colored_edges=[], colored_nodes=[]):
        """
        This method shows the current graph.
        """
        n_vertices = len(self.adjacency_list)
        matrix = np.zeros((n_vertices, n_vertices))
        key_to_index = dict(zip(self.adjacency_list.keys(), range(n_vertices)))
        for vertex, edges in self.adjacency_list.items():
            for edge in edges:
                matrix[key_to_index[vertex], key_to_index[edge]] = 1
        
        graph = nx.from_numpy_array(matrix, create_using=nx.Graph)
        pos = nx.nx_agraph.graphviz_layout(graph, prog="circo")
        nx.draw_networkx(graph,
                         pos=pos,
                         labels=dict(enumerate(self.adjacency_list.keys())),
                         with_labels=True,
                         node_size=500,
                         width=1.5,
                         node_color=['g' if node in colored_nodes else 'b' for node in self.adjacency_list],
                         edge_color=["r" if edge in colored_edges or (edge[1], edge[0]) in colored_edges else "k" for edge in graph.edges])
        nx.draw_networkx_edge_labels(graph,
                                     pos=pos,
                                     edge_labels={edge: str(self.adjacency_list[edge[0]][edge[1]]) for edge in graph.edges})
        plt.show()

############ CODE BLOCK 100 ################

class Dijkstra():
    """
    This call implements Dijkstra's algorithm and has at least the following object attributes after the object is called:
    Attributes:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the distances it takes to reach these nodes.
        :type priorityqueue: list[tuple[int, int]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that leads to this node and
                        the distance it takes to get to this node.
        :type history: dict[int, tuple[int, int]]
    """
    
    def __call__(self, graph, source, show_intermediate=True):
        """
        This method finds the fastest path from the source to all other nodes.

        :param graph: The graph on which the algorithm is used.
        :type graph: Graph
        :param source: The source node from which the fastest path needs to be found.
        :type source: int
        :param show_intermediate: This determines if intermediate results are shown.
                                  You do not have to do anything with the parameters as it is already programmed.
        :type show_intermediate: bool
        :return: A list of edges that make up all shortest paths
        :rtype: list[tuple[int]]
        """
        self.graph = graph
        self.show_intermediate = show_intermediate
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}

        self.main_loop()
        return self.find_shortest_edges()     

    def find_shortest_edges(self):
        """
        This method finds the shortest edge that creates the shortest paths between the source node and all other nodes.
        
        :return: A list of edge that form the optimal paths.
        :rtype: list[tuple[int]]
        """
        edges = []
        for node, (prev, _) in self.history.items():
            if prev is not None:
                edges.append((prev, node))
        return edges     

    def main_loop(self):
        """
        This method contains the logic for Dijkstra's algorithm

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        if self.show_intermediate:
            print("The green nodes indicate the current node that is explored.\nThe red edges indicate the current optimal edge.")
        
        while self.priorityqueue:
            # Get the node with the smallest distance
            self.priorityqueue.sort(key=lambda x: x[1])
            current_node, current_dist = self.priorityqueue.pop(0)

            for neighbor, weight in self.next_step(current_node):
                self.step(current_node, neighbor, weight)

            # This shows each step of Dijkstra's algorithm. 
            if self.show_intermediate:
                self.graph.show(self.find_shortest_edges(), [current_node])

    
    def step(self, node, new_node, weight):
        """
        One step in the Dijkstra algorithm. 

        :param node: The current node
        :type node: int
        :param new_node: The next node that can be visited from the current node
        :type new_node: int
        :param weight: The weight of the edge between the node and new_node
        :type weight: int
        """
        new_distance = self.history[node][1] + weight

        if new_node not in self.history or new_distance < self.history[new_node][1]:
            self.history[new_node] = (node, new_distance)
            self.priorityqueue.append((new_node, new_distance))
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: int
        :return: A list with possible next nodes and their weights that can be visited from the current node.
        :rtype: list[tuple[int, int]]
        """
        return list(self.graph[node].items())

############ CODE BLOCK 200 ################

class Prim():
    """
    This call implements Prim's algorithm and has at least the following object attributes after the object is called:
    Attributes:
        :param priorityqueue: The priority queue that is used to determine which node is explored next
        :type priorityqueue: list[tuple[int, int]]
        :param history: The history of nodes that are visited in the algorithm
        :type history: set[int]
        :param edges: A dictionary that contains which edge is kept
        :type edges: dict[int, int]
    """
    
    def __call__(self, graph, show_intermediate=True):
        """
        This method finds a minimal spanning tree.

        :param graph: The graph on which the algorithm is used.
        :type graph: Graph
        :param show_intermediate: This determines if intermediate results are shown.
                                  You do not have to do anything with the parameters as it is already programmed.
        :type show_intermediate: bool
        :return: A list of edges that make up the minimal spanning tree.
        :rtype: list[tuple[int]]
        """
        self.graph = graph
        self.show_intermediate = show_intermediate
        
        source = graph.get_random_node()
        self.priorityqueue = [(source, 0)]
        self.history = {source}
        self.edges = {}
        
        self.main_loop()
        return list(self.edges.items())    

    def main_loop(self):
        """
        This method contains the logic of Prim's Algorithm

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        if self.show_intermediate:
            print("All nodes that are in history are colored green.\nThe minimal edge are colored red, given the history.")
        
        while self.priorityqueue:
            # Get the node with the smallest edge weight
            self.priorityqueue.sort(key=lambda x: x[1])
            current_node, current_weight = self.priorityqueue.pop(0)

            for neighbor, weight in self.next_step(current_node):
                self.step(current_node, neighbor, weight)

            # This shows each step of Prim's algorithm. 
            if self.show_intermediate:
                self.graph.show(list(self.edges.items()), list(self.history))

    def step(self, node, new_node, new_weight):
        """
        One step in Prim's algorithm. 
        
        :param node: The current node
        :type node: int
        :param new_node: The next node that can be visited from the current node
        :type new_node: int
        :param new_weight: The weight of the edge between the node and new_node
        :type new_weight: int
        """
        if new_node not in self.history:
            self.history.add(new_node)
            self.edges[(node, new_node)] = new_weight
            self.priorityqueue.append((new_node, new_weight))

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: int
        :return: A list with possible next nodes and their weights that can be visited from the current node.
        :rtype: list[tuple[int, int]]
        """
        return list(self.graph[node].items())

############ CODE BLOCK 300 ################

class Forest():
    """
    This is a datastructure class for a forest of trees.
    It has the following attribute:
        :param trees: The trees in the forest.
        :type trees: list[set[int]]
    """
    def __init__(self, size):
        """
        This initializes the forest object, where the size of the forest is the number of trees
        and each tree consist of one element (in Kruskal's algorithm this would be a node).

        :param size: The size of the forest
        :type size: int
        """
        self.trees = [{i} for i in range(size)]

    def union_tree(self, tree1, tree2):
        """
        This method creates the union of two trees.

        :param tree1: One of the trees that needs to be merged.
        :type tree1: int
        :param tree2: One of the trees that needs to be merged.
        :type tree2: int
        """
        self.trees[tree1].update(self.trees[tree2])
        self.trees.pop(tree2)

    def find_tree(self, element):
        """
        This method finds to which tree a element belongs and
        returns the index of this tree.

        :param element: The element that we want to find (in Kruskal this would be node).
        :type element: int
        :return: The index of the tree in the forest list.
        :rtype: int
        """
        for i, tree in enumerate(self.trees):
            if element in tree:
                return i
        return None
            
class Kruskal():
    """
    This call implements Kruskal's algorithm and has at least the following object attributes after the object is called:
    Attributes:
        :param queue: The priority queue that is used to determine which edge is explored next
        :type queue: list[tuple[int]]
        :param forest: The forest for Kruskal's algorithm
        :type forest: Forest
        :param edges: A list of edges that contains which edge is kept
        :type edges: list[tuple[int]]
    """
    
    def __call__(self, graph, show_intermediate=True):
        """
        This method finds a minimal spanning tree.

        Note, that you need to use the attribute names given above.

        Hint: You might need more attributes then the list given above.

        :param graph: The graph on which the algorithm is used.
        :type graph: Graph
        :param show_intermediate: This determines if intermediate results are shown.
                                  You do not have to do anything with the parameters as it is already programmed.
        :type show_intermediate: bool
        :return: A list of edges that make up the minimal spanning tree.
        :rtype: list[tuple[int]]
        """
        self.show_intermediate = show_intermediate
        self.graph = graph
        self.queue = self.sort_edges()
        self.forest = self.create_forest()
        self.edges = []

        self.main_loop()
        return [(int(edge[0]), int(edge[1])) for edge in self.edges]

    def create_forest(self):
        """
        This method creates the initial forest of trees for Kruskal's algorithm given a graph.

        :return: A forest of trees each tree containing one node
        :rtype: Forest
        """
        return Forest(len(self.graph.adjacency_list))

    def sort_edges(self):
        """
        This method sorts the edges in ascending order from smallest weight to largest.
        
        Hint 1: For Kruskal's algorithm you only need one edge either from node A to B or from node B to A.
                So, you can essentially return the edges of any directed graph that corresponds to the undirect graph.
        Hint 2: You can sort a list using sorted with a function for each item. 
                For example, a function that lets you sort the list of tuples on the third item of each tuple.
                Have a look at the `key` argument of `sorted` if you want to know how to do this.

        :return: A list with all sorted edges without their weights.
        :rtype: list[tuple[int]]
        """
        edges = []
        for node, neighbors in self.graph.adjacency_list.items():
            for neighbor, weight in neighbors.items():
                if (neighbor, node) not in edges:
                    edges.append((node, neighbor, weight))
        return sorted(edges, key=lambda x: x[2])

    def main_loop(self):
        """
        This method contains the logic of Kruskal's Algorithm

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        if self.show_intermediate:
            print("The nodes of the current edge are colored green.\nThe current minimal edge are colored red.")

        while self.queue:
            edge = self.queue.pop(0)
            node1, node2, weight = edge
            tree1 = self.forest.find_tree(node1)
            tree2 = self.forest.find_tree(node2)

            if tree1 != tree2:
                self.edges.append((int(node1), int(node2)))
                self.forest.union_tree(tree1, tree2)

                if self.show_intermediate:
                    self.graph.show(self.edges, [node1, node2])

    def step(self, edge):
        """
        One step in Kruskal's algorithm. 
        
        :param edge: The current edge that we are exploring
        :type edge: tuple[int]
        """
        node1, node2, weight = edge
        tree1 = self.forest.find_tree(node1)
        tree2 = self.forest.find_tree(node2)

        if tree1 != tree2:
            self.edges.append((int(node1), int(node2)))  # Ensure the nodes are integers
            self.forest.union_tree(tree1, tree2)

############ CODE BLOCK 310 ################

class ForestFast():
    """
    This is a datastructure class for a forest of trees.
    It has the following attribute:
        :param trees: The trees in the forest (in Kruskal size is the number of nodes).
        :type trees: np.ndarray(int, (size,))
    """
    def __init__(self, size):
        """
        This initializes the forest object, where the size of the forest is the number of trees
        and each tree consists of one element (in Kruskal's algorithm this would be a node).

        Thi

        :param size: The size of the forest
        :type size: int
        """
        self.trees = np.arange(size, dtype=int)

    def union_tree(self, tree1, tree2):
        """
        This method creates the union of two trees.

        :param tree1: One of the trees that needs to be merged.
        :type tree1: int
        :param tree2: One of the trees that needs to be merged.
        :type tree2: int
        """
        self.trees[self.trees == tree2] = tree1

    def find_tree(self, element):
        """
        This method finds to which tree a element belongs and
        returns the index of this tree.

        :param element: The element that we want to find (in Kruskal this would be node).
        :type element: int
        :return: The index of the tree in the forest list.
        :rtype: int
        """
        return int(self.trees[element])

class KruskalFast(Kruskal):
    """
    This call implements Kruskal's algorithm and has at least the following object attributes after the object is called:
    Attributes:
        :param queue: The priority queue that is used to determine which edge is explored next
        :type queue: list[tuple[int]]
        :param forest: The forest for Kruskal's algorithm
        :type forest: ForestFast
        :param edges: A list of edges that contains which edge is kept
        :type edges: list[tuple[int]]
    """
    def create_forest(self):
        """
        This method creates the initial forest of trees for Kruskal's algorithm given a graph.

        :return: A forest of trees each tree containing one node
        :rtype: ForestFast
        """
        return ForestFast(len(self.graph.adjacency_list))


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
