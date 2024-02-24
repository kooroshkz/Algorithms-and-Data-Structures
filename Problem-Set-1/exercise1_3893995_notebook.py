############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

############ CODE BLOCK 1 ################
class Graph():
    """
    This is a class implementing a Graph.
    The internal representation of this class is an adjacency list.
    With set_adjacency_matrix and set_adjacency_list a graph can be retrieved with both representations.
    However, the internal representation stays an adjacency list.

    Attributes:
        :param self.adjacency_list: This contains a dict with as keys the name of the vertices
                               and as values their edges to other vertices.
        :type self.adjacency_list: dict[int/str, set[int/str]]
    """
    def __init__(self):
        """
        This initializes the graph.
        Note that this creates an empty graph.
        """
        self.adjacency_list = {}

    def add_vertex(self, vertex):
        """
        This adds a vertex to the graph.
        
        :param vertex: This is the name of the vertex
        :type vertex: int or str
        """
        if vertex in self.adjacency_list:
            raise ValueError(f"The vertex name {vertex} already exist")
            
        self.adjacency_list[vertex] = set()

    def add_edge(self, source, destination):
        """
        This adds an edge to the graph.

        :param source: This is the name of the vertex where the edge comes from.
        :type source: int or str
        :param destination: This is the name of the vertex where the edge goes to.
        :type destination: int or str
        """
        if source not in self.adjacency_list:
            raise ValueError(f"The vertex name {source} does not exist in the graph")
        if destination not in self.adjacency_list:
            raise ValueError(f"The vertex name {destination} does not exist in the graph")

        self.adjacency_list[source].add(destination)
        
    def set_adjacency_list(self, adjacency_list):
        """
        This uses the input to adjacency_list to set the graph.
        This overwrites the current graph.

        :param adjacency_list: This contains a dict with as keys the name of the vertices
                               and as values their edges to other vertices.
        :type adjacency_list: dict[int/str, set[int/str]]
        """
        # Check the new graph for consistency (no edges to non-existing vertices).
        for source, edges in adjacency_list.items():
            for dest in edges:
                if dest not in adjacency_list.keys():
                    raise ValueError(f"The vertex {dest} does not exist in the graph with vertices: {list(adjacency_list.keys())}")

            self.adjacency_list[source] = set(edges)

    def get_adjacency_list(self):
        """
        This method returns the adjacency list, which is the internal representation of the graph.

        :return: A dictionary containing the adjacency_list
        :rtype: dict[int/str, set[int/str]]
        """
        return self.adjacency_list

############ CODE BLOCK 2 ################
    def get_adjacency_matrix(self):
        """
        This method returns an adjacency matrix representation of the graph.

        :return: A 2D numpy array containing the adjacency matrix
        :rtype: np.ndarray[int]
        """
        return Graph.list_to_matrix(self.adjacency_list)

    @staticmethod
    def list_to_matrix(adjacency_list):
        """
        This static method transforms an adjacency list into an adjacency matrix.

        :param adjacency_list: An adjacency list representing a graph.
        :type adjacency_list: dict[int/str, set[int/str]]
        :return: A 2D numpy array containing the adjacency matrix
        :rtype: np.ndarray[bool]
        """

        vertices = list(adjacency_list.keys())
        vertices.sort()
        n = len(vertices)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        for i, vertex in enumerate(vertices):
            for edge in adjacency_list[vertex]:
                j = vertices.index(edge)
                adjacency_matrix[i, j] = 1
        return adjacency_matrix

    def show(self):
        """
        This method shows the current graph.
        """
        graph = nx.from_numpy_array(self.get_adjacency_matrix(), create_using=nx.DiGraph)
        nx.draw_shell(graph,
                      labels=dict(enumerate(self.adjacency_list.keys())),
                      with_labels=True,
                      node_size=500,
                      width=2,
                      arrowsize=20)
        plt.show()

############ CODE BLOCK 3 ################
    def set_adjacency_matrix(self, adjacency_matrix):
        """
        This class method expects an adjacency matrix as input, it then updates the Graph's internal
        adjacency list which updates the Graph. Note that this will overwrite the current Graph.

        :param adjacency_matrix: This contains the adjacency matrix.
        :type adjacency_matrix: np.ndarray[int]
        """
        adjacency_list = Graph.matrix_to_list(adjacency_matrix)
        self.set_adjacency_list(adjacency_list)

    @staticmethod
    def matrix_to_list(adjacency_matrix):
        """
        This static method transforms an adjacency matrix into an adjacency list.

        :param adjacency_matrix: A 2D numpy array containing the adjacency matrix
        :type adjacency_matrix: np.ndarray[int]
        :return: An adjacency list representing a graph.
        :rtype: dict[int, set[int]]
        """
        adjacency_list = {}
        for i, vertex in enumerate(adjacency_matrix):
            adjacency_list[i] = set(np.where(vertex == 1)[0])
        return adjacency_list

############ CODE BLOCK 4 ################
    def to_undirected_graph(self):
        """
        This method returns the undirected graph based on the direct graph (self).

        :return: This returns a graph object which is undirected.
        :rtype: Graph
        """
        undirected_graph = Graph()
        undirected_adjacency_list = copy.deepcopy(self.get_adjacency_list())
        for vertex, edges in undirected_adjacency_list.items():
            for edge in edges:
                undirected_adjacency_list[edge].add(vertex)
        undirected_graph.set_adjacency_list(undirected_adjacency_list)
        return undirected_graph


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
