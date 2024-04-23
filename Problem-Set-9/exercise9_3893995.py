############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 100 ################

def recursive_fibo(n):
    """
    This function calculates the nth Fibonacci sequence number.

    Note, that in this lab we will define the Fibonacci sequence as follows:
    $fibo(0) = 0$ and $fibo(1) = 1$
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    return recursive_fibo(n-1) + recursive_fibo(n-2)

############ CODE BLOCK 110 ################

class FibonacciBottomUp():
    """
    This class has one object attribute:
        :param fibo_numbers: A dictionary with the Fibonacci number, where the dictionary works like this: fibo(key) = value
        :type fibo_numbers: dict[int, int]
    """
    def __init__(self):
        """
        Here the Fibonacci numbers are initialized for the function because they are always the same.
        """
        self.fibo_numbers = {0: 0, 1: 1}

    def __call__(self, n):
        """
        This method calculates the nth Fibonacci number using bottom-up dynamic programming.

        :param n: The nth Fibonacci number
        :type n: int
        :return: fibo(n)
        :rtype: int
        """
        if n in self.fibo_numbers:
            return self.fibo_numbers[n]
        
        for i in range(2, n+1):
            self.fibo_numbers[i] = self.fibo_numbers[i-1] + self.fibo_numbers[i-2]
        return self.fibo_numbers[n]
            
    def step(self, n):
        """
        This calculates recursively the nth Fibonacci number.
        
        :param n: The nth Fibonacci number
        :type n: int
        :return: fibo(n)
        :rtype: int
        """
        return self(n)

############ CODE BLOCK 120 ################

class FibonacciTopDown(FibonacciBottomUp):
    def __call__(self, n):
        """
        This method calculates the nth Fibonacci number using top-down dynamic programming.

        :param n: The nth Fibonacci number
        :type n: int
        :return: fibo(n)
        :rtype: int
        """
        if n in self.fibo_numbers:
            return self.fibo_numbers[n]
        
        self.fibo_numbers[n] = self(n-1) + self(n-2)
        return self.fibo_numbers[n]

############ CODE BLOCK 130 ################

def fibonacci(max_):
    """
    The maximum Fibonacci number you want to calculate.

    Note that if max_ = 5 then fibo(max_) = fibo(5) = 5, in other words, max_ is the nth index starting from 0.

    :param max_: The maximum Fibonacci number you want to calculate, this includes max_.
    :type max_: int
    :return: The table with all Fibonacci numbers till `max_`
    :rtype: ndarray[int, (max_)]
    """
    fibo = np.zeros(max_+1, dtype=int)
    fibo[1] = 1
    for i in range(2, max_+1):
        fibo[i] = fibo[i-1] + fibo[i-2]
    return fibo

############ CODE BLOCK 140 ################

def recursive_fact(n):
    """
    This function calculates the nth factorial number.

    Note, that $fact(0) = 1$ and $fact(1) = 1$
    """
    if n == 0:
        return 1
    if n == 1:
        return 1
    return n * recursive_fact(n-1)

############ CODE BLOCK 150 ################

class Factorial():
    """
    This class has one object attribute:
        :param fact_numbers: A dictionary with the Factorial numbers, where the dictionary works like this: fact(key) = value
        :type fact_numbers: dict[int, int]
    """
    def __init__(self):
        """
        Here the Factorial numbers are initialized for the function because they are always the same.
        """
        self.fact_numbers = {0: 1}

    def __call__(self, n):
        """
        This method calculates the nth Factorial number using top-down dynamic programming.

        :param n: The nth Fibonacci number
        :type n: int
        :return: fact(n)
        :rtype: int
        """
        if n in self.fact_numbers:
            return self.fact_numbers[n]
        
        self.fact_numbers[n] = n * self(n-1)
        return self.fact_numbers[n]
            
    def step(self, n):
        """
        This calculates recursively the nth Factorial number.
        
        :param n: The nth Fibonacci number
        :type n: int
        :return: fact(n)
        :rtype: int
        """
        return self(n)

############ CODE BLOCK 200 ################

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

    def __repr__(self):
        """
        The representation of a Triangle object.
        It might be useful to create a (static) helper method.
        This should look as follows (for the example given above):
        
        [3]
        [7, 4]
        [2, 4, 6]
        [8, 5, 9, 3]

        Thus, each row is represented as a list and each row ends with a new line.
        """
        def helper(node):
            if node is None:
                return ""
            return f"[{node.value}] {helper(node.left)}{helper(node.right)}\n"

        return helper(self.top)

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

############ CODE BLOCK 210 ################

class MaximumPathSum():
    """
    This class has the following object attributes when the object are called:
        :parem partial_path_sum: The dynamic programming memory, here the maximum sum and the corresponding path (next node) for each node are saved.
        :type partial_path_sum: dict[Node, tuple[Node, int]]
    """
    
    def __call__(self, triangle):
        """
        This method calculated the maximum sum path in the triangle.

        :param triangle: The triangle for which you need to calculate the maximum path sum
        :type triangle: Triangle
        :return: The sum of the path from the top node of the triangle and the path that gives this sum
        :rtype: int, list[Node]
        """
        self.partial_path_sum = {}
        max_sum = self.step(triangle.top)
        path = self.find_path(triangle.top)
        return max_sum, path

    def step(self, node):
        """
        One divide and conquer step to calculate the maximum path sum

        :param node: The current node
        :type node: Node
        :return: The sum of the path from this node
        :rtype: int
        """
        if node is None:
            return 0
        if node in self.partial_path_sum:
            return self.partial_path_sum[node][1]

        left_sum = self.step(node.left)
        right_sum = self.step(node.right)
        max_child_sum = max(left_sum, right_sum)
        total_sum = node.value + max_child_sum
        self.partial_path_sum[node] = (None, total_sum)

        return total_sum

    def find_path(self, node):
        """
        Find the path with the highest sum.

        :param node: The starting node of the path, top of the triangle
        :type node: Node
        :return: The path that gives the largest sum
        :rtype: list[Node]
        """
        path = []
        while node is not None:
            path.append(node)
            node = self.partial_path_sum[node][0]
        return path[::-1]

############ CODE BLOCK 300 ################

class Primes():
    """
    This class has the following object attributes:
        :param primes: The list of prime numbers.
        :type primes: list[int]
    """
    def __init__(self):
        """
        Prime numbers do not change. So, we can use the same "memory" every time we run the callable object we initiate here.
        """
        self.primes = [2, 3]

    def __call__(self, max_):
        """
        Calculate all prime numbers up to the `max_` number

        :param max_: The largest number to check if it is prime.
        :param max_: int
        """
        for i in range(self.primes[-1]+2, max_+1, 2):
            is_prime = True
            for prime in self.primes:
                if i % prime == 0:
                    is_prime = False
                    break
            if is_prime:
                self.primes.append(i)
        return self.primes
                
    def __repr__(self):
        """
        Representation of the class objects, you can change it however you like.
        """
        return str(self.primes)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
