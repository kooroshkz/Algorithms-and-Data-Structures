############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
class MergeSort():
    def __call__(self, list_):
        """
        This method sorts a list and returns the sorted list.
        Note, that if two elements are equal the order should not change. 
        This is also known as the sorting algorithm is stable.

        :param list_: An unsorted list that needs to be sorted.
        :type list_: list[int/float]
        :return: The sorted list.
        :rtype: list[int/float]
        """
        return self.step(list_)

    def step(self, list_):
        """
        One step in the merge sort algorithm.
        Here, you split the list sort them both, and then merge them.

        :param list_: An unsorted list that needs to be sorted.
        :type list_: list[int/float]
        :return: The sorted list.
        :rtype: list[int/float]
        """
        if len(list_) <= 1:
            return list_
        mid = len(list_) // 2
        left = self.step(list_[:mid])
        right = self.step(list_[mid:])
        return self.merge(left, right)

    @staticmethod
    def merge(list1, list2):
        """
        This method merges two sorted lists into one sorted list.

        :param list1: A sorted list that needs to be merged.
        :type list1: list[int/float]
        :param list2: A sorted list that needs to be merged.
        :type list2: list[int/float]
        :return: The sorted list.
        :rtype: list[int/float]
        """
        result = []
        i = j = 0
        while i < len(list1) and j < len(list2):
            if list1[i] <= list2[j]:
                result.append(list1[i])
                i += 1
            else:
                result.append(list2[j])
                j += 1
        result.extend(list1[i:])
        result.extend(list2[j:])
        return result
        

############ CODE BLOCK 20 ################
def factorial_recursion(n):
    """
    This function calculates the nth factorial recursively.

    :param n: The nth factorial number
    :type n: int
    :return: n!
    :type: int
    """
    if n == 0:
        return 1
    return n * factorial_recursion(n - 1)

############ CODE BLOCK 30 ################
class BinarySearch():
    """
    A binary search class that can be used to make a callable object 
    which given a list and a value returns the index of the value.

    After __call__ the object has two attributes:
        :param list: A sorted list with values.
        :type list: list
        :param value: The value that you are searching for.
        :type value: int
    """
    def __call__(self, list_, value):
        """
        This method finds the index of a value in a list
        if a list does not have the value you should return None.

        :param list_: A sorted list with values.
        :type list_: list[int]
        :param value: The value that you are searching for.
        :type value: int
        :return: index of the found value.
        :rtype: int
        """
        self.list = list_
        self.value = value
        return self.step(0, len(list_))
    
    def step(self, min_index, max_index):
        """
        This is one step in the binary search algorithm.
        No helper methods are given but if you want you can create
        for example a next_step method or base_case method.

        :param min_index: The left index of your search space, thus the minimum value of your search space.
        :type min_index: int
        :param max_index: The right index of your search space, thus the maximum value of your search space.
        type max_index: int
        :return: index of the found value.
        :rtype: int
        """
        if min_index >= max_index:
            return None
        mid = (min_index + max_index) // 2
        if self.list[mid] == self.value:
            return mid
        if self.list[mid] < self.value:
            return self.step(mid + 1, max_index)
        return self.step(min_index, mid)

############ CODE BLOCK 40 ################
def gcd(a, b):
    """
    This function calculates the greatest common divisor of a and b.
    """
    if b == 0:
        return a
    return gcd(b, a % b)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
