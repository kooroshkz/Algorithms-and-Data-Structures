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
class Node():
    """
    This class creates node objects which can be used to build any kind of Tree 
    that has at most two children and the nodes contain values.

    Attributes:
        :param self.info: The value of the node.
        :type self.info: int
        :param self.left: The left child of this node, defaults to None.
        :type self.left: Node, optional 
        :param self.left: The right child of this node, defaults to None.
        :type self.left: Node, optional 
    """
    def __init__(self, info, left=None, right=None):
        self.info = info
        self.left = left
        self.right = right

    def __repr__(self):
        """
        This returns a representation of a Node object.

        :return: A string representing the Node object.
        :rtype: str
        """
        # Change this to anything you like, such that you can easily print a Node object.
        return super(Node, self).__repr__()

############ CODE BLOCK 20 ################
class BinaryTree():
    """
    This class creates binary tree objects.
    The tree itself is stored recursively. 
    This means that you can only access the root node directly, 
    while other nodes can be accessed through the root node. 
    (Later in this exercise you will implement several of these methods)

    Attributes:
        :param self.root: The root node of the binary tree, defaults to None.
        :type self.root: Node, optional
    """
    def __init__(self, root=None):
        """
        This initializes a binary tree object.
        Note, that this creates by default an empty tree.
        """
        self.root = root

    def add(self, value):
        """    
        Adds a new node to the binary tree, this should be placed randomly at any empty leaf node.
    
        :param value: The value to be added.
        :type value: int
        """
        if self.root is None:
            self.root = Node(value)
        else:
            self._add(self.root, value)

    def show(self, show_compact=False):
        """
        This method shows the tree, where the root node is colored blue, 
        the left nodes are colored green, and the right nodes are colored red.

        :param show_compact: This gives the option to show a compact form of the tree. 
                             This means that if a node has only one child it will be shown below it instead of left or right.
                             This parameter defaults to False
        :type show_compact: boolean, optional
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
            else:
                color_map.append("blue")
            
            if node.left is not None:
                color_map.append("green")
                add_node_edge(G, color_map, node_name, node.left)
            elif node.right is not None and not show_compact:
                G.add_node(f"N{node_name}")
                G.add_edge(node_name, f"N{node_name}")
                
            if node.right is not None:
                color_map.append("red")
                add_node_edge(G, color_map, node_name, node.right)
            elif node.left is not None and not show_compact:
                G.add_node(f"N{node_name}")
                G.add_edge(node_name, f"N{node_name}")
        
        # Make the graph
        G = nx.DiGraph()
        color_map = []
        add_node_edge(G, color_map, None, self.root)
        name_root = self.root.info

        # Generate the node positions
        pos = hierarchy_pos(G, root=self.root.info, leaf_vs_root_factor=1)
        new_pos = {k:v for k,v in pos.items() if str(k)[0] != 'N'}
        k = G.subgraph(new_pos.keys())

        if isinstance(self, BinarySearchTree) or isinstance(self, BinarySearchTreeDouble):
            nx.draw(k, pos=new_pos, node_color=color_map, with_labels=True, node_size=600)
        else:
            nx.draw(k, pos=new_pos, with_labels=True, node_size=600)

        # Set the plot settings
        x, y = zip(*pos.values())
        x_min, x_max = min(x), max(x)
        plt.xlim(1.01*x_min-0.01*x_max, 1.01*x_max-0.01*x_min)
        plt.ylim(min(y)-0.05, max(y)+0.05)
        plt.show()

############ CODE BLOCK 40 ################
class BinarySearchTree(BinaryTree):
    """
    This class creates binary tree objects.
    The tree itself is stored recursively. 
    This means that you can only access the root node directly, 
    while other nodes can be accessed through the root node. 
    (Later in this exercise you will implement several of these methods)

    Attributes:
        :param self.root: The root node of the binary tree, defaults to None.
        :type self.root: Node, optional
    """
    def search(self, value):
        """
        Returns a Tuple with the following two items:
         - the parent of the node with a certain value
         - the node with a certain value
        Note that having access to the parent will prove useful in
        other functions, such as adding and removing.
    
        If the tree does not contain the value, return
        the parent of the node where it should have been
        placed, and a None value.
    
        :param value: The value that you are searching.
        :param type: int
        :return: Tuple of the nodes as described above.
        :rtype: Node, Node
        """
        parent = None
        current = self.root
        while current is not None:
            if value == current.info:
                return parent, current
            elif value < current.info:
                parent = current
                current = current.left
            else:
                parent = current
                current = current.right
        return parent, None
    
    def add(self, value):
        """    
        Adds a new node to the binary search tree, respecting the condition that
        for each node, all values in the left sub-tree are smaller than its value,
        and all values in the right subtree are greater than its value.
        Only add the node with the value, if it does not exist yet in the tree.
    
        :param value: The value to be added.
        :type value: int
        """
        parent, _ = self.search(value)
        if parent is None:
            self.root = Node(value)
        elif value < parent.info:
            parent.left = Node(value)
        elif value > parent.info:
            parent.right = Node(value)

############ CODE BLOCK 60 ################
class BinarySearchTreeDouble(BinaryTree):
    """
    This is needed for the expert test part 3.0, which is optional and can be ignored for now.
    However, this cell must be run, otherwise, the show method breaks!!!
    """
    pass


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
