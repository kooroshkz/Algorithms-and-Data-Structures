############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################

class EqualSubsetSum():
    def __call__(self, list_):
        """
        This method tries to find a way to split the list into two lists of equal sum.
        If it is not possible to do this return None.

        :param list_: The original list with the values
        :type list_: list[int]
        :return: Two list of equal sum or None.
        :rtype: list[int], list[int]
        """
        return self.step(list_, [], [])

    def step(self, list_, list1, list2):
        """
        One Divide & Conquer step for the equal subset sum problem.

        :param list_: The original list with the values
        :type list_: list[int]
        :return: Two list of equal sum or None.
        :rtype: list[int], list[int]
        """
        if len(list_) == 0:
            if sum(list1) == sum(list2):
                return list1, list2
            else:
                return None
        else:
            return self.step(list_[1:], list1 + [list_[0]], list2) or self.step(list_[1:], list1, list2 + [list_[0]])

############ CODE BLOCK 15 ################

class AllEqualSubsetSum():
    def __call__(self, list_):
        """
        This method tries to find all ways to split the list into two lists of equal sum.
        If it is not possible to do this return an empty list.

        :param list_: The original list with the values
        :type list_: list[int]
        :return: A list containing tuples with two lists of equal sum.
        :rtype: list[tuple[list[int], list[int]]]
        """
        return self.step(list_, [], [])

    def step(self, list_, list1, list2):
        """
        One Divide & Conquer step for the equal subset sum problem.

        :param list_: The original list with the values
        :type list_: list[int]
        :return: Two list of equal sum or None.
        :rtype: list[int], list[int]
        """
        if len(list_) == 0:
            if sum(list1) == sum(list2):
                return [(list1, list2)]
            else:
                return []
        else:
            return self.step(list_[1:], list1 + [list_[0]], list2) + self.step(list_[1:], list1, list2 + [list_[0]])

############ CODE BLOCK 20 ################

class CoinChange():
    coins = [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    def __call__(self, amount):
        """
        One step in the divide and conquer algorithm.

        :param leftover_amount: The leftover amount of change. This is the original amount minus the change.
        :type leftover_amount: float
        :param change: A list of coins.
        :type change: list[float]
        :param max_coin_id: The index of the largest coin that this step can use.
        :type max_coin_id: int
        """
        return self.step(amount, [], 0)
        
    def step(self, leftover_amount, change, max_coin_id):
        """
        One step in the divide and conquer algorithm.

        :param leftover_amount: The leftover amount of change. This is the original amount minus the change.
        :type leftover_amount: float
        :param change: A list of coins.
        :type change: list[float]
        :param max_coin_id: The index of the largest coin that this step can use.
        :type max_coin_id: int
        """
        if leftover_amount == 0:
            return [change]
        if leftover_amount < 0 or max_coin_id >= len(self.coins):
            return []
        return self.step(leftover_amount - self.coins[max_coin_id], change + [self.coins[max_coin_id]], max_coin_id) + self.step(leftover_amount, change, max_coin_id + 1)

############ CODE BLOCK 30 ################

class Node():
    def __init__(self, position, next_nodes=None):
        """
        This node class has a position as indicated by the image above and
        it has several next nodes which are all nodes with position[0] + 1.

        :param position: a tuple with the position as indicate by the image.
        :type position: tuple[int]
        :param next_nodes: The next possible nodes with their weights 
                           which are the cost going from this node to the next.
        :type next_nodes: list[tuple[Node, float]]
        """
        self.position = position
        if next_nodes is None:
            self.next_nodes = []
        else:
            self.next_nodes = next_nodes

    def __repr__(self):
        """
        A representation of the node object.
        """
        return f"Node{self.position}"

class Graph():
    """
    A graph is an object that contains all nodes to find the fastest path between two nodes.
    It has the following attributes:
        :param start: The first node in the graph
        :type start: Node
        :param end: The last node in the graph
        :type end: Node
        :param nodes: A list of all the nodes in the graph.
        type nodes: list[nodes]
    """
    def __init__(self, generate=True, min_size=3, max_size=10):
        """
        This method creates a random graph containing nodes.
        It also sets the next_nodes for each node.

        :param min_size: minimum width of the graph.
        :type min_size: int
        :param max_size: maximum width of the graph.
        :type max_size: int
        """
        self.start = None
        self.end = None
        self.nodes = []
        if generate:
            self.generate_random_graph(min_size, max_size)

    def generate_random_graph(self, min_size, max_size):
        """
        Generate a random graph.
        """
        n = RNG.integers(min_size, max_size)

        self.start = Node((0,0))
        self.end = Node((n+1,0))
        self.nodes = [self.start, self.end]
        previous_nodes = [self.start]
        for i in range(1, n+2):
            if i <= n:
                new_nodes = [Node((i,j)) for j in range(max(2, int(RNG.normal(5, 2))))]
                self.nodes.extend(new_nodes)
            else:
                new_nodes = [self.end]
                
            for prev_node in previous_nodes:
                weights = list(RNG.integers(1,10, len(new_nodes)))
                prev_node.next_nodes = list(zip(new_nodes, weights))

            previous_nodes = new_nodes

        self.__create_adjacency_matrix()
    
    def __create_adjacency_matrix(self):
        """
        This is an internal method to generate the adjacency_matrix for the graph.
        """
        # This is just for printing
        self.__adjacency_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node in enumerate(self.nodes):
            for next_node, weight in node.next_nodes:
                try:
                    self.__adjacency_matrix[i, self.nodes.index(next_node)] = weight
                except ValueError:  # The sub graph has ended
                    pass
                    

    def split_graph(self, node1=None, node2=None):
        """
        This method splits the graph into 3 separate graphs.
        Note, that nodes are not copied for each of the graphs.

        :param node1: A node that the path must go through.
        :type node1: Node
        :param node2: Another node that the path must go through.
        :type node2: Node
        :return: The three "new" graphs with "old" nodes.
        :rtype: list[graph]
        """
        # Determine node b and c
        if node1 is None and node2 is not None:
            node1 = node2
            node2 = None
        
        if node1 is None:
            node1 = RNG.choice(self.nodes[2:])
        if node2 is None:
            nodes = [node for node in self.nodes[2:] if node.position[0] != node1.position[0]]
            node2 = RNG.choice(nodes)
        b, c = (node1, node2) if node1.position[0] < node2.position[0] else (node2, node1)

        # divide the graph into separate graphs
        # set start and end
        graphs = [Graph(False) for _ in range(3)]
        for i, node in enumerate([self.start, b, c, self.end]):
            if i > 0:
                graphs[i-1].nodes.append(node)
                graphs[i-1].end = node
            if i < 3:
                graphs[i].nodes.append(node)
                graphs[i].start = node

        # set all nodes in between.
        graph_id = 0 
        for node in self.nodes[2:]:
            if node.position[0] < graphs[graph_id].end.position[0]:
                graphs[graph_id].nodes.append(node)
            elif node.position[0] > graphs[graph_id].end.position[0]:
                graph_id += 1
                # test if this new node should be added to the next graph
                if node.position[0] < graphs[graph_id].end.position[0]:
                    graphs[graph_id].nodes.append(node)

        # set adjacency_matrix
        for graph in graphs:
            graph.__create_adjacency_matrix()

        return graphs
    
    def show(self, path=None):
        """
        This method shows the current graph.
        """
        graph = nx.from_numpy_array(self.__adjacency_matrix, create_using=nx.DiGraph)
        middle = {self.start.position[0]: self.start.position[1], 
                  self.end.position[0]: self.end.position[1]}
        node = self.start
        for n in range(node.position[0], self.end.position[0]-1):
            middle[n+1] = len(node.next_nodes) // 2
            node = node.next_nodes[0][0]
            
        pos = {i: (node.position[0], node.position[1] - middle[node.position[0]]) for i, node in enumerate(self.nodes)}
        edge_labels = nx.get_edge_attributes(graph, "weight")
        color_map = []
        for node1, node2 in graph.edges:
            if path is not None and self.nodes[node1] in path and self.nodes[node2] in path:
                color_map.append("r")
            else:
                color_map.append("k")
        nx.draw_networkx(graph, 
                         pos,
                         labels = {i: f"{node.position[0]},{node.position[1]}" for i, node in enumerate(self.nodes)},
                         edge_color= color_map,
                         with_labels=True,
                         node_size=800,
                         width=1,
                         arrowsize=15)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=0.3, font_size=8)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=0.7, font_size=8)
        plt.show()

############ CODE BLOCK 32 ################

class SolveFastestPath():
    def __call__(self, graph, source, destination):
        """
        This method solve the problem of the shortest path between the source and destination.
        This can be done using a breadth-first search algorithm.

        :param graph: The graph in which you need to find the path.
        :type graph: Graph
        :param source: The start node from which the path begins.
        :type source: Node
        :param destination: The end node where the path ends.
        :type destination: Node
        :return: This returns a path and the cost of a path.
                 Note, that a path contains the source and destination node.
        :rtype: list[node], float
        """
        # Initialize a queue for BFS
        queue = [(source, [source])]
        visited = set()

        while queue:
            # Dequeue a node and its path
            node, path = queue.pop(0)
            # Check if the node has already been visited
            if node not in visited:
                # Mark the node as visited
                visited.add(node)
                # Check if we've reached the destination
                if node == destination:
                    return path, self.calculate_path_cost(path)
                # Enqueue all adjacent nodes
                for next_node, _ in node.next_nodes:
                    queue.append((next_node, path + [next_node]))
        # If no path is found
        return None, float('inf')

    def calculate_path_cost(self, path):
        """
        Calculate the total cost of a path in the graph.

        :param path: The path to calculate the cost for.
        :type path: list[node]
        :return: The total cost of the path.
        :rtype: float
        """
        cost = 0
        for i in range(len(path) - 1):
            for next_node, weight in path[i].next_nodes:
                if next_node == path[i + 1]:
                    cost += weight
                    break
        return cost

############ CODE BLOCK 35 ################

def shortest_path_via_nodes(graph, node1=None, node2=None):
    """
    This function has as input a "full" graph and as output the cost
    of going from the start node to the end node through node1 and node2.

    If node1 and/or node2 are none the algorithm chooses a random node to go through.

    :param graph: The graph in which you need to find the path.
    :type graph: Graph
    :param node1: A node that the path must go through.
    :type node1: Node
    :param node2: Another node that the path must go through.
    :type node2: Node
    :return: This returns a path, the cost of a path, and the node that it needs to go through.
             Note, that a path contains the source and destination node.
    :rtype: list[node], float, node1, node2 
    """
    graphs = graph.split_graph(node1, node2)
    paths = [SolveFastestPath()(g, g.start, g.end) for g in graphs]
    total_cost = sum(path[1] for path in paths)
    return [p for path in paths for p in path[0]], total_cost, node1, node2


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
