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


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
