############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 100 ################

class Tribonacci():
    """
    This class has one object attribute:
        :param tribo_numbers: A dictionary with the Tribonacci number, where the dictionary works like this: tri(key) = value
        :type tribo_numbers: dict[int, int]
    """
    def __init__(self):
        """
        Here the Tribonacci numbers are initialized for the function because they are always the same.
        See, the object attributes described above.
        """
        self.tribo_numbers = {0: 0, 1: 0, 2: 1}

    def __call__(self, n):
        """
        This method calculates the nth Tribonacci number using dynamic programming.

        :param n: The nth Tribonacci number
        :type n: int
        :return: tri(n)
        :rtype: int
        """
        if n in self.tribo_numbers:
            return self.tribo_numbers[n]
        else:
            for i in range(3, n+1):
                self.tribo_numbers[i] = self.tribo_numbers[i-1] + self.tribo_numbers[i-2] + self.tribo_numbers[i-3]
            return self.tribo_numbers[n]
            
    def step(self, n):
        """
        This calculates recursively the nth Tribonacci number.
        
        :param n: The nth Tribonacci number
        :type n: int
        :return: tri(n)
        :rtype: int
        """
        if n in self.tribo_numbers:
            return self.tribo_numbers[n]
        else:
            return self.step(n-1) + self.step(n-2) + self.step(n-3)

############ CODE BLOCK 200 ################

class CoinChange():
    """
    This class has one class attribute:
        :param coins: This is a list containing all coins
        :type coins: list[int]
        
    This class has one object attribute:
        :param coin_change: A dictionary, where the keys are the change amount and max coin id, and the values are a list containing all combinations to give the change back limited by the max coin. 
        :type coin_change: dict[tuple[int], list[list[int]]]
    """
    coins = [200, 100, 50, 20, 10, 5, 2, 1]

    def __init__(self):
        self.coin_change = {(0, c): [[]] for c in range(len(CoinChange.coins))}
    
    def __call__(self, amount):
        """
        This method calculates all possible ways to give change for the amount.

        :param amount: The amount that is the sum of the change.
        :type amount: int
        :return: A list with all possible ways to give change. The change consists of a list of coins.
        :rtype: list[list[int]]
        """
        return self.step(amount, len(CoinChange.coins)-1)

    def step(self, amount, max_coin_id):
        """
        One step in the divide and conquer algorithm.

        :param leftover_amount: The leftover amount of change. This is the original amount minus the change.
        :type leftover_amount: int
        :param change: A list of coins.
        :type change: list[int]
        :param max_coin_id: The index of the largest coin that this step can use.
        :type max_coin_id: int
        :return: A list with all possible ways to give change. The change consists of a list of coins.
        :rtype: list[list[int]]
        """
        if (amount, max_coin_id) in self.coin_change:
            return self.coin_change[(amount, max_coin_id)]
        else:
            change = []
            for i in range(max_coin_id, -1, -1):
                coin = CoinChange.coins[i]
                if amount == coin:
                    change.append([coin])
                elif amount > coin:
                    for c in self.step(amount-coin, i):
                        change.append([coin] + c)
            self.coin_change[(amount, max_coin_id)] = change
            return change

############ CODE BLOCK 300 ################

class Collatz():
    """
    This class has one object attribute:
        :param collatz_number: Think about how to store the intermediate results
        :type collatz_number: dict[...]
    """
    def __init__(self):
        self.collatz_number = {1: 1}


    def __call__(self, max_):
        """
        This method finds the starting number below `max_` with the longest sequence.

        Note, that if two starting numbers have the same sequence length the smallest number is the tie breaker.

        :param max_: The maximum starting number
        :type max_: int
        :return: The starting number with the longest sequence and the length of this sequence.
        :rtype: int, int
        """
        max_number = 1
        max_length = 1
        for i in range(1, max_):
            length = self.step(i)
            if length > max_length:
                max_number = i
                max_length = length
        return max_number, max_length

    def step(self, number):
        """
        This method calculates the next number in the sequences and returns its length.

        :param number: The current number
        :type number: int
        :return: The length of the sequences
        :rtype: int
        """
        if number in self.collatz_number:
            return self.collatz_number[number]
        else:
            if number % 2 == 0:
                self.collatz_number[number] = 1 + self.step(number // 2)
            else:
                self.collatz_number[number] = 1 + self.step(3 * number + 1)
            return self.collatz_number[number]

############ CODE BLOCK 400 ################

def knapsack1(max_weights, items):
    """
    This function returns the optimal value you can fit in the knapsack given the max weight and the items.
    The optimal value is returned by returning the filled-in dynamic programming table.

    :param max_weights: The maximum weight that is allowed in the knapsack.
    :type max_weights: int
    :param items: The items set you can choose from. Each item has a weight and value represented by a (weight, value) pair.
    :type items: list[tuple[int]]
    :return: The filled-in dynamic programming table.
    :rtype: ndarray(int, (max_weights+1, len(items)+1))
    """ 
    table = np.zeros((max_weights+1, len(items)+1), dtype=int)
    for i in range(1, len(items)+1):
        for w in range(1, max_weights+1):
            weight, value = items[i-1]
            if weight > w:
                table[w, i] = table[w, i-1]
            else:
                table[w, i] = max(table[w, i-1], value + table[w-weight, i-1])
    return table

############ CODE BLOCK 410 ################

def knapsack2(max_weights, items):
    """
    This function returns the optimal value you can fit in the knapsack given the max weight and the items.
    The optimal value is returned by returning the filled-in dynamic programming table and 
    the dictionary containing which items you need to fill the knapsack optimally.

    :param max_weights: The maximum weight that is allowed in the knapsack.
    :type max_weights: int
    :param items: The items set you can choose from. Each item has a weight and value represented by a (weight, value) pair.
    :type items: list[tuple[int]]
    :return: The filled-in table for the value and a table (dictionary) that tells you if an item is used or not for a certain subset and weight.
    :rtype: ndarray(int, (max_weights+1, len(items)+1)), dict[tuple[int], list[int]]
    """ 
    is_used_table = {(0,i) : [] for i in range(max_weights+1)}
    is_used_table |= {(i,0) : [] for i in range(len(items)+1)}
    
    table = np.zeros((max_weights+1, len(items)+1), dtype=int)
    
    # Your knapsack2 algorithm implementation here
    
    return table, is_used_table

############ CODE BLOCK 510 ################

def rod_cutting(sell_prices, rod_size=5):
    """
    This function calculates how a rod with a certain size can best be sold.
    This can be either as a whole or in (several) slices.

    :param sell_prices: The price per rod length
    :type sell_prices: list[int]
    :param rod_size: The size of the rod uncut.
    :type rod_size: int
    :return: The maximum value for which this rod can be sold.
    :rtype: int
    """
    current_prices = 0
    # Loop through all possible combinations of cutting the rod.
    for cuts in product(range(rod_size+1), repeat=rod_size):
        # Check if this is a possible way of cutting the rod
        if sum(cuts) == rod_size:
            total_sell_price = sum(sell_prices[cut] for cut in cuts)
            current_prices = max(current_prices, total_sell_price)
            
    return current_prices
                
class RodCuttingDivideAndConquer():
    def __call__(self, sell_prices, rod_size=5):
        """
        This method calculates how a rod with a certain size can best be sold.
        This can be either as a whole or in (several) slices.
    
        :param sell_prices: The price per rod length
        :type sell_prices: list[int]
        :param rod_size: The size of the rod uncut.
        :type rod_size: int
        :return: The maximum value for which this rod can be sold.
        :rtype: int
        """
        self.sell_prices = sell_prices
        return self.step(rod_size, rod_size)

    def step(self, rod_size, max_cutsize):
        """
        This method recursively calculates how a rod with a certain size can best be sold,
        given a maximum cut size.

        :param rod_size: The size of the rod uncut.
        :type rod_size: int
        :param max_cutsize: The maximum size of the rod that can be cut off.
        :type max_cutsize: int
        :return: The maximum value for which this rod can be sold.
        :rtype: int
        """
        if rod_size == 0:
            return 0
        current_prices = 0
        for cutsize in range(1, max_cutsize+1):
            if rod_size - cutsize >= 0:
                current_prices = max(current_prices, self.sell_prices[cutsize-1] + self.step(rod_size-cutsize, cutsize))
        return current_prices

############ CODE BLOCK 520 ################

class RodCuttingDynamicProgramming():
    def __call__(self, sell_prices, rod_size=5):
        """
        This method calculates how a rod with a certain size can best be sold.
        This can be either as a whole or in (several) slices.
    
        :param sell_prices: The price per rod length
        :type sell_prices: list[int]
        :param rod_size: The size of the rod uncut.
        :type rod_size: int
        :return: The maximum value for which this rod can be sold.
        :rtype: int
        """
        self.sell_prices = sell_prices
        return self.step(rod_size, rod_size)

    def step(self, rod_size, max_cutsize):
        """
        This method recursively calculates how a rod with a certain size can best be sold,
        given a maximum cut size.

        :param rod_size: The size of the rod uncut.
        :type rod_size: int
        :param max_cutsize: The maximum size of the rod that can be cut off.
        :type max_cutsize: int
        :return: The maximum value for which this rod can be sold.
        :rtype: int
        """
        table = np.zeros((rod_size+1, max_cutsize+1), dtype=int)
        for i in range(1, rod_size+1):
            for j in range(1, max_cutsize+1):
                if i >= j:
                    table[i, j] = max(table[i, j-1], self.sell_prices[j-1] + table[i-j, j])
                else:
                    table[i, j] = table[i, j-1]
        return table[rod_size, max_cutsize]

############ CODE BLOCK 530 ################

def rod_cutting_dynamic_programming(sell_prices, rod_size=5):
    """
    This function calculates how a rod with a certain size can best be sold.
    This can be either as a whole or in (several) slices.

    Use bottom-up dynamic programming to find the answer.

    :param sell_prices: The price per rod length
    :type sell_prices: list[int]
    :param rod_size: The size of the rod uncut.
    :type rod_size: int
    :return: The maximum value for which this rod can be sold.
    :rtype: int
    """
    table = np.zeros((rod_size+1, rod_size+1), dtype=int)
    for i in range(1, rod_size+1):
        for j in range(1, rod_size+1):
            if i >= j:
                table[i, j] = max(table[i, j-1], sell_prices[j-1] + table[i-j, j])
            else:
                table[i, j] = table[i, j-1]
    return table[rod_size, rod_size]


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
