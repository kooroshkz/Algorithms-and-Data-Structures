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
            current = self.root
            while True:
                if value <= current.info:
                    if current.left is None:
                        current.left = Node(value)
                        break
                    else:
                        current = current.left
                else:
                    if current.right is None:
                        current.right = Node(value)
                        break
                    else:
                        current = current.right

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

############ CODE BLOCK 21 ################
    @staticmethod
    def get_highest_subtree_value(subtree):
        """
        Gets the highest value out of a subsection of a binary tree (not a binary search tree!!).
        This should be a recursive method.

        :param subtree: The node of the subtree.
        :type subtree: Node
        :return: The highest value within this subtree.
        :rtype: int
        """
        if subtree is None:
            return float('-inf')
        else:
            return max(subtree.info, BinaryTree.get_highest_subtree_value(subtree.left), BinaryTree.get_highest_subtree_value(subtree.right))
        
    def get_highest_value(self):
        """
        Gets the highest value out of a binary tree (not a binary search tree!!).
        This function should use the recursive static method `_get_highest_subtree_value`.

        :return: The highest value within this tree.
        :rtype: int
        """
        return BinaryTree.get_highest_subtree_value(self.root)
        
    @staticmethod
    def count_subtree_leafs(subtree):
        """
        Counts the number of leafs in a subsection of a binary tree.
        This should be a recursive method.
    
        :param subtree: The root of the subtree.
        :type subtree: Node
        :return: The number of leafs in the subtree.
        :rtype: int
        """
        if subtree is None:
            return 0
        elif subtree.left is None and subtree.right is None:
            return 1
        else:
            return BinaryTree.count_subtree_leafs(subtree.left) + BinaryTree.count_subtree_leafs(subtree.right)


    def count_leafs(self):
        """
        Counts the number of leafs in a binary tree.
        This method should use the recursive method `count_subtree_leafs`.
            
        :return: The number of leafs in the tree.
        :rtype: int
        """
        return BinaryTree.count_subtree_leafs(self.root)

    @staticmethod
    def get_subtree_height(subtree):
        """
        Determines the height of a subsection of a binary tree. 
        This should be a recursive method.
    
        :param subtree: The root of the subtree.
        :type subtree: Node
        :return: The height of the subtree.
        :rtype: int
        """
        if subtree is None:
            return 0
        else:
            return 1 + max(BinaryTree.get_subtree_height(subtree.left), BinaryTree.get_subtree_height(subtree.right))


    def get_height(self):
        """
        Determines the height of a binary tree.
        This method should use the recursive method `get_subtree_height`.

        :return: The height of the tree.
        :rtype: int
        """
        return BinaryTree.get_subtree_height(self.root)

############ CODE BLOCK 22 ################
    def __repr__(self):
        """
        This returns the representation of a tree. See above, for how this should look like.
        """
        if self.root is None:
            return "Empty Tree"

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

############ CODE BLOCK 41 ################
    def get_highest_value(self, subtree=None):
        """
        Gets the highest value out of a binary search tree.
        Implement a faster algorithm compared to a general binary tree highest value algorithm given that this is a binary search tree.
        If no arguments are given this method should give the highest value of the whole tree.

        Note, that an empty tree has a negative infinite value as the highest value.
        
        :param subtree: The root of the subtree, defaults to None.
        :type subtree: Node
        :return: The highest value within this (sub)tree.
        :rtype: int
        """
        if subtree is None:
            subtree = self.root
        if subtree is None:
            return float('-inf')
        elif subtree.right is None:
            return subtree.info
        else:
            return self.get_highest_value(subtree.right)

############ CODE BLOCK 42 ################
    def get_lowest_value(self, subtree=None):
        """
        Gets the lowest value out of a binary search tree.
        Implement a faster algorithm compared to a general binary tree lowest value algorithm given that this is a binary search tree.
        If no arguments are given this method should give the lowest value of the whole tree.

        Note, that an empty tree has a positive infinite value as the highest value.

        :param subtree: The root of the subtree, defaults to None.
        :type subtree: Node
        :return: The lowest value within this (sub)tree.
        :rtype: int
        """
        if subtree is None:
            subtree = self.root
        if subtree is None:
            return float('inf')
        elif subtree.left is None:
            return subtree.info
        else:
            return self.get_lowest_value(subtree.left)

    def is_binary_search_tree(self, subtree=None):
        """
        Returns whether the tree is a valid binary search tree. 
        Hint: You can implement this as a recursive method. 
        You can under some assumptions make use of
        get_lowest_value and get_highest_value.
        If no arguments are given this method should check if the whole tree is a valid binary search tree.

        :param subtree: The root of the subtree, defaults to None.
        :type subtree: Node
        :return: true iff it is a valid binary search tree
        :rtype: Boolean
        """
        if subtree is None:
            subtree = self.root
        if subtree is None:
            return True
        
        if subtree.left is None:
            left_valid = True
        else:
            left_valid = subtree.info > self.get_lowest_value(subtree.left) and self.is_binary_search_tree(subtree.left)
            
        if subtree.right is None:
            right_valid = True
        else:
            right_valid = subtree.info < self.get_highest_value(subtree.right) and self.is_binary_search_tree(subtree.right)
        
        return left_valid and right_valid

############ CODE BLOCK 43 ################
    def remove(self, value):
        """
        Removes a node from the binary search tree. 
    
        :param value: The value that needs to be deleted.
        :type value: int
        """

        parent, node = self.search(value)
        
        if node is None:
            return  # Node not found, nothing to remove
        
        if node.left is None and node.right is None:
            # Case 1: Node is a leaf node
            if parent is None:
                self.root = None  # Node is root and tree has only one node
            elif parent.left == node:
                parent.left = None
            else:
                parent.right = None
                
        elif node.left is None or node.right is None:
            # Case 2: Node has only one child
            if node.left is None:
                child = node.right
            else:
                child = node.left
                
            if parent is None:
                self.root = child  # Node is root
            elif parent.left == node:
                parent.left = child
            else:
                parent.right = child
                
        else:
            # Case 3: Node has two children
            successor_parent = node
            successor = node.right
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
                
            node.info = successor.info
            
            if successor_parent.left == successor:
                successor_parent.left = successor.right
            else:
                successor_parent.right = successor.right

############ CODE BLOCK 50 ################
class NodeD():
    """
    This class creates node objects which can be used to build any kind of Tree 
    that has at most two children and the nodes contain values.

    Attributes:
        :param self.info: The value of the node.
        :type self.info: int
        :param self.parent: The parent of this node, defaults to None.
        :type self.parent: Node, optional 
        :param self.left: The left child of this node, defaults to None.
        :type self.left: Node, optional 
        :param self.left: The right child of this node, defaults to None.
        :type self.left: Node, optional 
    """
    def __init__(self, info, parent=None, left=None, right=None):
        self.info = info
        self.parent = parent
        self.left = left
        self.right = right

    def __repr__(self):
        """
        This returns a representation of a Node object.

        :return: A string representing the Node object.
        :rtype: str
        """
        # Change this to anything you like, such that you can easily print a Node object.
        return super(NodeD, self).__repr__() 

############ CODE BLOCK 60 ################
class BinarySearchTreeDouble(BinaryTree):
    """
    This class creates binary tree objects with double-linked nodes (NodeD).
    The tree itself is stored recursively. 
    This means that you can only access the root node directly, 
    while other nodes can be accessed through the root node. 

    Attributes:
        :param self.root: The root node of the binary tree, defaults to None.
        :type self.root: Node, optional
    """
    def __init__(self, root=None):
        """
        This initializes a binary tree object.
        Note, that this creates an empty tree
        """
        self.root = root
    
    def search(self, value):
        """
        Returns a NodeD with the value that was searched for.
        Note, that NodeD objects already have access to the parent.
        So, it does not need to be returned.
    
        If the tree does not contain the value, return None.
    
        :param value: The value that you are searching.
        :param type: int
        :return: The NodeD with attribute info equal to value
        :rtype: NodeD
        """
        current = self.root
        while current is not None:
            if value == current.info:
                return current
            elif value < current.info:
                current = current.left
            else:
                current = current.right
        return None

    def add(self, value):
        """    
        Adds a new node to the binary search tree, respecting the condition that
        for each node, all values in the left sub-tree are smaller than its value,
        and all values in the right subtree are greater than its value.
        Only add the node with the value, if it does not exist yet in the tree.
    
        :param value: The value to be added.
        :type value: int
        """
        if self.root is None:
            self.root = NodeD(value)
        else:
            self._add(self.root, value)

    def remove(self, value):
        """
        Removes a node from the binary search tree. 
    
        :param value: The value that needs to be deleted.
        :type value: int
        """
        node_to_remove = self.search(value)
        
        if node_to_remove is None:
            return  # Node not found, nothing to remove
        
        parent = node_to_remove.parent
        
        if node_to_remove.left is None and node_to_remove.right is None:
            # Case 1: Node is a leaf node
            if parent is None:
                self.root = None  # Node is root and tree has only one node
            elif parent.left == node_to_remove:
                parent.left = None
            else:
                parent.right = None
                
        elif node_to_remove.left is None or node_to_remove.right is None:
            # Case 2: Node has only one child
            if node_to_remove.left is None:
                child = node_to_remove.right
            else:
                child = node_to_remove.left
                
            if parent is None:
                self.root = child  # Node is root
            elif parent.left == node_to_remove:
                parent.left = child
            else:
                parent.right = child
                
        else:
            # Case 3: Node has two children
            successor_parent = node_to_remove
            successor = node_to_remove.right
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
                
            node_to_remove.info = successor.info
            
            if successor_parent.left == successor:
                successor_parent.left = successor.right
            else:
                successor_parent.right = successor.right


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
