############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
class Permutations():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating all state-space but 
        using backtracking to ignore incorrect solution/solution branches.
        
        :param list_: The list containing all the unique elements.
        :type list_: list
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # All actions are the same, so it is helpful to make an object attribute.
        self.list = list_

        return self.step([])

    def step(self, chosen_list):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.
        
        :param chosen_list: This list contains all objects chosen so far.
        :type chosen_list: list[Objects]
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        # If the permutation is correct, we can return it.
        if len(chosen_list) == len(self.list):
            return [chosen_list]

        # If the permutation is incorrect, we can return an empty list.
        if not self.is_not_incorrect(chosen_list):
            return []

        # If the permutation is not incorrect, we can generate the next step.
        return self.next_step(chosen_list, self.list)
        
    def next_step(self, chosen_list, chosen_object):
        """
        This method generates the actions that are possible for the next step and calls step with the updated state-space.
        These actions consist of all elements of the original list. 
        Note, that if you saved them as an object attribute this method does not do much.

        :param i: A counter how many elements are added to the new permutation.
        :type i: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        # We need to make a copy of the list, so we can remove the object from the list.
        chosen_object = copy.deepcopy(chosen_object)
        permutations = []
        
        # We iterate over all objects in the list.
        for obj in chosen_object:
            # We remove the object from the list.
            chosen_object.remove(obj)
            # We add the object to the chosen list.
            new_chosen_list = copy.deepcopy(chosen_list)
            new_chosen_list.append(obj)
            # We call the next step with the new state-space.
            permutations += self.step(new_chosen_list)
            # We add the object back to the list.
            chosen_object.append(obj)
        
        return permutations
    
    def is_not_incorrect(self, chosen_list):
        """
        This method returns if the state-space is (partially) correct aka if it can become a permutation.

        :param chosen_list: A possible permutation of self.list
        :type chosen_list: list[Objects]
        :return: Return if the permutation variable is or is not a permutation.
        :rtype: boolean
        """
        # We check if the list is a permutation.
        return len(set(chosen_list)) == len(chosen_list)

############ CODE BLOCK 20 ################
def constraint(queens, col):
    """
    The constraints for the n-queen problem

    :param queens: The currently placed queens.
    :type queens: list[int]
    :param col: The column that the queen is placed in.
    :type col: int
    :return: If the constraint is satisfied or not.
    :rtype: bool
    """
    row = len(queens)
    for r, c in enumerate(queens):
        if c == col or r - c == row - col or r + c == row + col:
            return False
    return True
       

def rec_nQueens(size, queens=None):
    """
    Recursively computes a solution for the n-Queens puzzle.

    :param size: The size of the puzzle
    :type size: int
    :param queens: The currently placed queens, e.g. [4,2] represent 
                   that on row 0 we placed a queen in the 4th position, 
                   and on row 1 we placed a queen in the 2nd position.
                   This defaults to [].
    :type queens: list[int], optional
    :return: The (partial) list of queen positions.
    :rtype: list[int]
    """
    if queens is None:
        queens = []
    
    if size <= len(queens):
        return queens
    
    for col in range(size):
        if constraint(queens, col):
            queens.append(col)
            candidate_sol = rec_nQueens(size, queens)
            if candidate_sol:
                return candidate_sol
            queens.pop()
    return False

class N_Queens():
    def __call__(self, size):
        """
        Recursively computes a solution for the n-Queens puzzle.
        
        size is not a recursive part of n-queens.
        So, we can store it in an object attribute.
        
        :param size:  The size of the puzzle
        :type size: int
        """
        self.size = size 
        return self.step([])

    def step(self, queens):
        """
        One step in solving the n-queens problem

        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :return: The (partial) list of queen positions.
        :rtype: list[int]
        """
        if self.size <= len(queens):
            return queens
        
        for col in range(self.size):
            candidate_sol = self.next_step(queens, col)
            if candidate_sol:
                return candidate_sol
                
        self.clean_up(queens)
        return False
        
    def next_step(self, queens, col):
        """
        Check if you can go to the next step.
        
        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :param col: The column that is tried to be added.
        :type col: int
        :return: The return of the step method
        :rtype: list[int]
        """
        if self.constraint(queens, col):
            queens.append(col)
            return self.step(queens)
        return False

    def clean_up(self, queens):
        """
        Clean up your previous division, in this case, remove the last queen of the board.
        """
        queens.pop()
    
    def constraint(self, queens, col):
        """
        Check if the constraints are satisfied.

        :param queens: The currently placed queens, e.g. [4,2] represent 
                       that on row 0 we placed a queen in the 4th position, 
                       and on row 1 we placed a queen in the 2nd position.
        :type queens: list[int]
        :param col: The column that is tried to be added.
        :type col: int
        :return: The return of the step method
        :rtype: list[int]
        """
        row = len(queens)
        for r, c in enumerate(queens):
            if c == col or r - c == row - col or r + c == row + col:
                return False
        return True 


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
