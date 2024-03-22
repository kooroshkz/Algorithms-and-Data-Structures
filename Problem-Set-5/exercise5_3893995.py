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
def distance(pointA, pointB):
    """
    This calculates the Euclidean distance between two points: https://en.wikipedia.org/wiki/Euclidean_distance

    :param pointA: The first coordinate
    :type pointA: list[float] or np.ndarray[(2,), float]
    :param pointB: The second coordinate
    :type pointB: list[float] or np.ndarray[(2,), float]
    :return: The distance between the two points
    :rtype: float
    """
    return np.linalg.norm(np.array(pointA) - np.array(pointB))

def nearest_neighbour(data, point):
    """
    This function finds the nearest neighbour of "point" in the "data".

    :param data: All the points (neighbours) that need to be compared to "point".
    :type data: np.ndarray[(n, 2), float]
    :param point: The point of which you want to find the closest neighbour.
    :type point: list[float] or np.ndarray[(2,), float]
    :return: The nearest neighbour and the distance to that neighbour.
    :rtype: tuple[np.ndarray[(2,), float], float]
    """
    return min([(neighbour, distance(neighbour, point)) for neighbour in data if not np.array_equal(neighbour, point)], key=lambda x: x[1])

############ CODE BLOCK 15 ################
def classify_point(data, point):
    """
    This function finds the nearest group based on the nearest neighbour of each group.
    Use `nearest_neighbour` in this function.

    :param data: A list of groups, where each group consists of all the points (neighbours) that need to be compared to "point".
    :type data: list[np.ndarray[(Any, 2), float]]
    :param point: The point of which you want to find the closest group.
    :type point: list[float] or np.ndarray[(2,), float]
    :return: The nearest group and the nearest neighbour.
    :rtype: tuple[int, np.ndarray[(Any, 2), float]]
    """
    return min([(i, nearest_neighbour(group, point)) for i, group in enumerate(data)], key=lambda x: x[1][1])

############ CODE BLOCK 20 ################
class Ring():
    """
    One ring of the door.
    """
    def __init__(self, current=0, size=3):
        """
        Attributes:
            :param current: Current value that is on top of the ring.
            :type current: int
            :param size: The number of options that this ring has.
            :type size: int
            :param answer: the correct position of this ring.
            :type answer: int            
        """
        self.__current = current
        self.__size = size
        self.__answer = RNG.integers(size)

    def turn(self):
        """
        This method turns the ring clockwise and
        returns if the ring is in the original order.

        :return: if the ring is in the original order.
        :rtype: boolean
        """
        self.__current = (self.__current + 1) % self.__size
        return not self.__current

    def correct(self):
        """
        This method check if the ring is currently in the right position.
        """
        return self.__current == self.__answer
        
    def __repr__(self):
        return f"Ring({chr(self.__current + 65)})"

class DoorRingPuzzle():
    """
    This class represents a door with a certain amount of rings 
    that need to be positioned in the correct order to open the door.
    """
    def __init__(self, n_rings=None, size=3):
        """
        This initialized the door. 
        
        Attributes:
            :param rings: The rings of the door.
            :type rings: list[Ring}


        :param n_rings: The number of rings this door has, defaults to 3
        :type n_rings: int, optional
        :param size: The size of each ring (number of options), defaults to 3
                     This can also be a list with the size of each individual ring.
                     This list should have the same length as n_rings.
        :type size: list[int] or int, optional
        """
        if not isinstance(size, int):
            if n_rings != len(size) and not n_rings is None:
                raise ValueError("The number of rings should be equal to the number of sizes that are given for each individual ring!")
            self.__rings = [Ring(0, s) for s in size]            
        else:
            if n_rings is None:
                n_rings = 3
            self.__rings = [Ring(0, size) for _ in range(n_rings)]

    def turn_ring(self, ring):
        """
        This method can rotate one ring clockwise.
        It also tells the user if the ring is back in its original position.
        Thus with the "A" on top.

        :param ring: The ring that is rotated.
        :type ring: int
        :return: If the ring is in its original position.
        :rtype: boolean
        """
        return self.__rings[ring].turn()

    def open_door(self):
        """
        This method checks if you can open the door.

        :return: If opening the door succeeded.
        :rtype: boolean
        """
        for ring in self.__rings:
            if not ring.correct():
                return False
        return True

    def __len__(self):
        """
        This gives the length of the door which
        is defined as the number of rings.
        """
        return len(self.__rings)

    def __repr__(self):
        return str(self.__rings)    

############ CODE BLOCK 25 ################
class SolveDoorRingPuzzle():
    def __call__(self, door):
        """
        This method solves the door with "n" rings problem.
        You do not need to return anything because of side effects.
        See, this exercise of ITP on an explanation of side effects:
        https://joshhug.github.io/LeidenITP/labs/lab7/#exercise-1-shopping-list-standard

        :param door: The door that needs to be opened.
        :type door: DoorRingPuzzle
        """
        self.door = door
    
    def step(self, ring):
        """
        This is one step in the exhaustive search algorithm which uses depth-first search.

        :param ring: The ring that is currently turned.
        :type ring: int
        :return: If the door is opened in the current configuration.
        :rtype: boolean
        """
        if ring == len(self.door):
            return self.door.open_door()
        if self.door.turn_ring(ring):
            return self.step(ring + 1)
        return self.step(ring)
            
    def next_step(self, ring):
        """
        This determines the next step in the exhaustive search.
    
        :param ring: The ring that is currently turned.
        :type ring: int
        :return: This method returns what self.step returns
        :type: boolean
        """
        return self.step(ring)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
