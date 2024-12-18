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
        idx=[]
        for x in adjacency_list:
            idx.append(x)
        n = len(adjacency_list)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        for i in adjacency_list:
            for j in adjacency_list[i]:
                adjacency_matrix[idx.index(i)][idx.index(j)]=1
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
        n=len(adjacency_matrix)
        adjacency_list={}
        for i in range(0, n):
            adjacency_list[i]=set()
            for j in range(0, n):
                if adjacency_matrix[i][j]==1:
                    adjacency_list[i].add(j)
        return adjacency_list

############ CODE BLOCK 4 ################
    def to_undirected_graph(self):
        """
        This method returns the undirected graph based on the direct graph (self).

        :return: This returns a graph object which is undirected.
        :rtype: Graph
        """
        g = Graph()
        g.adjacency_list = self.adjacency_list
        for i in g.adjacency_list:
            for j in g.adjacency_list[i]:
                g.adjacency_list[j].add(i)
        return g

############ CODE BLOCK 5 ################
    def has_two_or_less_odd_vertices(self):
        """
        This method determines if an undirected graph has at most two vertices with an odd number of neighbours.
        
        Any graph can be used but should first be converted into a undirected graph, 
        which is already given in the code below.
        
        Note that a reflexive edge, i.e. an edge that connects a vertex to itself, should be ignored. 
        For example the partial adjacency list: `4: {4, 2, 1}` has an even number of neighbours.
        
        :return: True if graph contains at most two vertices with odd number of neighbours, False otherwise.
        :rtype: bool
        """
        g = self.to_undirected_graph()
        counter = 0
        for i in self.adjacency_list:
            neighbour_sum = len(self.adjacency_list[i])
            if i in self.adjacency_list[i]:
                neighbour_sum-=1
            if neighbour_sum%2!=0:
                counter+=1
        if counter<=2:
            return True
        return False

############ CODE BLOCK 6 ################
    def invert_edges(self):
        """
        This method inverts all edges of a graph.

        Note, that inverting the edges of an undirected graph returns the same graph.
        
        :return: A new graph with the edge inverted.
        :rtype: Graph
        """
        g = Graph()
        for i in self.adjacency_list:
            g.adjacency_list[i] = set()
        for i in self.adjacency_list:
            for j in self.adjacency_list[i]:
                g.adjacency_list[j].add(i) 
        return g


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
