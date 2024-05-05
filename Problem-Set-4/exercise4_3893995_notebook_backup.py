############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
def alternating_disks(n):
    """
    This function solves the alternating disks problem for a row with size 2*n containing n light disks and n dark disks.
    The function returns an array with n ones (light disks) and n zeros (dark disks).
    Use the function swap to change the position of two disks.
    
    :param n: Number of light or dark disks
    :type n: int
    :return: The ordered list (light is 1 and dark is 0)
    :rtpye: np.ndarray[(2*n), int]
    """
    disks = np.array(list(zip(np.ones(n), np.zeros(n)))).flatten()
    return disks

def swap(disks, i):
    """
    This function swaps the disks i and i+1 in the array disks.
    This is a helper function for alternating_disks.

    :param disks: Array containing the light and dark disks
    :type disks: np.ndarray[(2*n), int]
    :param i: Position of the disk that needs to be swapped
    :type i: int
    """
    pass

############ CODE BLOCK 20 ################
class Permutations():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating them first using step and
        then filtering them using `self.is_correct`.

        :param list_: The list containing all the elements.
        :type list_: list[Objects]
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # Nothing needs to be initialized or filter. So just call self.step
        return self.step(list_)
        
    def step(self, list_):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.

        :param list_: The list containing all the possible elements (that are valid actions)
        :type list_: list
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        if len(list_) == 0:
            return [[]]

        all_permutations = []
        for i, elem in enumerate(list_):
            next_permutations = self.next_step(list_[:i] + list_[i+1:], i)
            for perm in next_permutations:
                all_permutations.append([elem] + perm)
        return all_permutations

    def next_step(self, list_, index):
        """
        This method generates the actions that are possible for the next step and calls step.
        These actions consist of the elements of the list that are not yet in the new permutated list.

        :param list_: The list containing all the possible elements (that are valid actions)
                      from the previous state-space.
        :type list_: list
        :param index: The index of the element that is used as action in the previous step.
        :type index: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        return self.step(list_)

############ CODE BLOCK 25 ################
class PermutationsWithReplacement():
    def __call__(self, list_):
        """
        This method gives all the permutations of the list.
        This is done by generating them first using step and
        then filtering them using `self.is_correct`.
        
        :param list_: The list containing all the unique elements.
        :type list_: list
        :return: A list with all possible permutations.
        :rtype: list[list[Objects]]
        """
        # all actions are the same, so it is helpful to make an object attribute.
        self.list = list_
        # filter out all incorrect state-spaces and return all permutations of self.list
        return [x for x in self.step(0) if self.is_correct(x)]

    def step(self, i):
        """
        This method adds one value to the new permutated list and
        and calls next_step to generate a new set of actions.
        
        :param i: A counter how many elements are added to the new permutation.
        :type i: int
        :return: A list containing all the permutations, from the current space-state.
        :type: list[list[Objects]]
        """
        if i == len(self.list):
            return [[]]

        all_permutations = []
        for elem in self.list:
            next_permutations = self.next_step(i)
            for perm in next_permutations:
                all_permutations.append([elem] + perm)
        return all_permutations

    def next_step(self, i):
        """
        This method generates the actions that are possible for the next step and calls step.
        These actions consist of all elements of the original list.

        :param i: An counter how many elements are added to the new permutation.
        :type i: int
        :return: This method returns what self.step returns
        :type: list[list[Objects]]
        """
        return self.step(i + 1)
    
    def is_correct(self, permutation):
        """
        This method returns if the state-space is correct aka if it is a permutation.

        :param permutation: A possible permutation of self.list
        :type permutation: list[Objects]
        :return: Return if the permutation variable is or is not a permutation.
        :rtype: boolean
        """
        return len(set(permutation)) == len(self.list)

############ CODE BLOCK 30 ################
class ManhattanProblem():
    def __call__(self, road_grid):
        """
        This method gives all the fastest routes through this part of Manhattan.
        You always start top left and end bottom right.
        This is done by calling step which should return a list of routes, 
        where a route consists of a list of coordinates.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :return: A list with all possible routes, where a route consists of a list of coordinates.
        :rtype: list[list[tuple[int]]]
        """
        self.grid = road_grid
        return self.next_step((0,0))  # We already are in the first state-space so we need to generate the next actions.
        
    def step(self, pos, actions):
        """
        This method does one step in the depth-first search in the Manhattan grid.
        One step consists of adding one coordinate (tuple) to the route and
        generating all possible routes from this coordinate in the grid,
        this is done recursively.

        :param pos: The current coordinate in the grid.
        :type pos: tuple[int]
        :param actions: List of possible next coordinates.
        :type actions: list[tuple[int]]
        :return: All possible route with this position as starting point.
        :rtype: list[list[tuple[int]]]
        """
        if pos == (self.grid.shape[0]-1, self.grid.shape[1]-1):
            return [[pos]]
        
        all_routes = []
        for action in actions:
            next_routes = self.next_step(action)
            for route in next_routes:
                all_routes.append([pos] + route)
        return all_routes

    def next_step(self, pos):
        """
        Here, we check which actions we can take depending on the current position in the grid.
        Then, we call next step with the current position and next possible actions.

        :param pos: The current coordinate in the grid.
        :type pos: tuple[int]
        :return: This method returns what self.step returns
        :rtype: list[list[tuple[int]]]
        """
        actions = self.get_actions(pos)
        return self.step(pos, actions)
        


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
