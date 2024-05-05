############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 100 ################

def fair_splitting_basic(list_):
    """
    This function solves the fair splitting problem by looping through the `list_`.

    Note that the value should be added to the second list if the sum of both lists is equal.

    :param list_: A list with values
    :type list_: list[int]
    :return:: Two lists with a similar sum.
    :rtype: tuple[list[int]]
    """
    list_ = sorted(list_, reverse=True)
    list1 = []
    list2 = []
    for i in list_:
        if sum(list1) <= sum(list2):
            list1.append(i)
        else:
            list2.append(i)
    return list1, list2

############ CODE BLOCK 110 ################

def fair_splitting_sorted(list_):
    """
    This function solves the fair splitting problem by first sorting the `list_`.

    Note that the value should be added to the second list if the sum of both lists is equal.

    :param list_: A list with values
    :type list_: list[int]
    :return:: Two lists with a similar sum.
    :rtype: tuple[list[int]]
    """
    list_ = sorted(list_, reverse=True)
    list1 = []
    list2 = []
    for i in list_:
        if sum(list1) <= sum(list2):
            list1.append(i)
        else:
            list2.append(i)
    return list1, list2

############ CODE BLOCK 200 ################

def flashlight_problem(crossing_time):
    """
    This function solves the flashlight problem and calculates the total time it takes to cross the bridge.

    :param crossing_time: A list of the time it takes each person to cross the bridge.
    :type crossing_time: list[int]
    :return: The total time it takes for the group to cross the bridge
    :rtype: int
    """

    crossing_time.sort()
    total_time = 0

    while len(crossing_time) > 3:
        # Two fastest cross, first fastest returns, two slowest cross, second fastest returns
        total_time += crossing_time[1] * 2 + crossing_time[0] + crossing_time[-1]
        crossing_time = crossing_time[:-2]  # remove two slowest

    if len(crossing_time) == 3:
        total_time += crossing_time[0] + crossing_time[1] + crossing_time[2]
    elif len(crossing_time) == 2:
        total_time += crossing_time[1]
    else:  # len(crossing_time) == 1
        total_time += crossing_time[0]

    return total_time

############ CODE BLOCK 300 ################

class Node():
    def __init__(self, value, left=None, right=None):
        """
        This creates a node object containing the value and its lower right and left node.

        :param value: The value of the node
        :type value: int
        :param left: The lower left node, defaults to None
        :type left: Node
        :param right: The lower right
        :type right: Node        
        """
        self.value = value
        self.left = left
        self.right = right

    def __lt__(self, other):
        """
        This makes the use of lesser than and higher than between nodes possible.
        """
        return self.value < other.value

    def __repr__(self):
        """
        This is the representation of the Node.
        You can adjust it to your own preference.
        """
        return f"Node({self.value})"

class Triangle():
    """
    This class has one object attribute:
        :param top: The top node of the triangle
        :type top: Node
    """
    
    def __init__(self, min_=1, max_=100, height=10):
        """
        This creates a random triangle object as explained above.

        :param min: The minimum value that can be generated as value for a node
        :type min: int
        :param max: The maximum value that can be generated as value for a node
        :type max: int
        :param depth: The height of the triangle.
        :type depth: int
        """
        # Initialize all nodes
        rows = []
        for row in range(1, height+1):
            rows.append([Node(RNG.integers(min_, max_)) for _ in range(row)])

        # Connect all nodes
        for i, row in enumerate(rows[:-1]):
            for j, cell in enumerate(row):
                cell.right = rows[i+1][j+1]
                cell.left  = rows[i+1][j]

        self.top = rows[0][0]
        self.height = height

    def show(self, path=None):
        """
        This method shows the current triangle.
        """
        graph = nx.Graph()

        pos = {}
        row = [self.top]
        for i in range(self.height):
            for j, node in enumerate(row):
                if node.left is not None:
                    graph.add_edge(node, node.left)
                    graph.add_edge(node, node.right)
                    
                graph.add_node(node)
                pos[node] = (j * 2 + (self.height-i), -i)
                
            row = [row[0].left] + [n.right for n in row]
            
        nx.draw_networkx(graph, 
                         pos,
                         labels = {node: node.value for node in pos},
                         node_color="w",
                         with_labels=True,
                         node_size=600,
                         width=1)
        
        if path is not None:
            edgelist = [(node, path[i+1]) for i, node in enumerate(path[:-1])]
            nx.draw_networkx_edges(graph, 
                                   pos, 
                                   edgelist,
                                   edge_color='r',
                                   width=4)
        plt.show()

############ CODE BLOCK 310 ################

def greedy_path_sum(triangle):
    """
    This function finds the maximum sum path in the triangle using a greedy algorithm.

    Note that you always take the right node if the left node and right node are equal.

    :param triangle: The triangle for which you need to calculate the maximum path sum
    :type triangle: Triangle
    :return: The sum of the path from the top node of the triangle and the path that gives this sum
    :rtype: int, list[Node]
    """
    path = [triangle.top]
    total = triangle.top.value
    current = triangle.top

    while current.left is not None:
        if current.left.value == current.right.value:
            path.append(current.right)
            total += current.right.value
            current = current.right
        elif current.left.value > current.right.value:
            path.append(current.left)
            total += current.left.value
            current = current.left
        else:
            path.append(current.right)
            total += current.right.value
            current = current.right

    return total, path

############ CODE BLOCK 400 ################

def knapsack_problem(max_weight, items):
    """
    This function solves the knapsack problem and returns the value you can fit in the knapsack given the max weight and the items.
    It also returns which items lead to this value.

    :param max_weight: The maximum weight that is allowed in the knapsack.
    :type max_weight: int
    :param items: The items set you can choose from. Each item has a weight and value represented by a (weight, value) pair.
    :type items: list[tuple[int]]
    :return: The total value in the knapsack and the items that are added (by index).
    :rtype: int, list[int]
    """ 
    n = len(items)
    K = np.zeros((n + 1, max_weight + 1))

    for i in range(1, n + 1):
        for w in range(1, max_weight + 1):
            weight, value = items[i - 1]
            if weight <= w:
                K[i][w] = max(value + K[i - 1][w - weight], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    result = K[n][max_weight]
    total_weight = max_weight
    taken = []

    for i in range(n, 0, -1):
        if result <= 0:
            break
        if result == K[i - 1][total_weight]:
            continue
        else:
            taken.append(i - 1)
            result -= items[i - 1][1]
            total_weight -= items[i - 1][0]

    return int(K[n][max_weight]), taken

############ CODE BLOCK 500 ################

class IntegerSet():
    def __init__(self, M):
        """
        This initialized the hashtable. This should include making an array of the proper length and object type.

        :param M: The number of buckets in the hashtable.
        :type size: int
        """        
        self.M = M
        self.hash_table = np.empty(M, dtype=object)  # make an empty array of type object. This can now be any Python object
        self.hash_table[...] = [[] for _ in range(M)]  # fill each element of the array with an empty lists

    def get_bucket_number(self, value):
        """
        This is the function that determines which bucket to use.

        :param value: The integer for which we want to know the bucket number. The result is just the value mod the number of buckets.
        :type value: int
        :return: The index in the hash table of the value.
        :rtype: int
        """
        return value % self.M
    
    def add(self, value):
        """
        This method adds a value to the IntegerSet.
        Note, that the value should be placed at the right index.
        If the index already contains elements then it should be added to the end of the list.

        Important, because we are making a set, you should check if the value is already in the list of the correct index.
        If so, you should not add it again.
        
        :param value: The integer that we want to add to the set.
        :type value: int
        """
        bucket_number = self.get_bucket_number(value)
        if value not in self.hash_table[bucket_number]:
            self.hash_table[bucket_number].append(value)

    def __contains__(self, value):
        """
        This method checks if a value is in a list.
        This magic/dunder method is called in the following syntax: `value in integer_set`, 
        where integer_set is an object of this class.
        
        Note, that this is essentially a search method.

        :param value: The integer that we are searching in the set.
        :type value: int
        :return: This returns if the value is found or not
        :rtype: bool
        """
        bucket_number = self.get_bucket_number(value)
        return value in self.hash_table[bucket_number]

    def remove(self, value):
        """
        This method removes value from the set.

        :param value: The integer that we are deleting from the set.
        :type value: int
        """
        bucket_number = self.get_bucket_number(value)
        if value in self.hash_table[bucket_number]:
            self.hash_table[bucket_number].remove(value)
        
    def __repr__(self):
        """
        Representation of the IntegerSet.
        Currently, it shows the hash table, where each row is an index in the table.
        You can change it to anything you like, for example, you could represent it as a set of integers.
        """
        return "[" + ",\n ".join(map(repr, self.hash_table)) + "]"

############ CODE BLOCK 510 ################

class Set(IntegerSet):
    def get_bucket_number(self, item):
        """
        This is the hash function that is used to calculate the index in the hash table given a value.

        :param item: The hashable object for which we want to know the index in the hash table.
        :type item: object 
        :return: The index in the hash table of the item of type object.
        :rtype: int
        """
        return hash(item) % self.M


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
