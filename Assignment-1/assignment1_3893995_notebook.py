############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
class Sudoku():
    """
    This class creates sudoku objects which can be used to solve sudokus. 
    A sudoku object can be any size grid, as long as the square root of the size is a whole integer.
    To indicate that a cell in the sudoku grid is empty we use a zero.
    A sudoku object is initialized with an empty grid of a certain size.

    Attributes:
        :param self.grid: The sudoku grid containing all the digits.
        :type self.grid: np.ndarray[(Any, Any), int]  # The first type hint is the shape, and the second one is the dtype. 
        :param self.size: The width/height of the sudoku grid.
        :type self.size: int
    """
    def __init__(self, size=9):
        self.grid = np.zeros((size, size))
        self.size = size
        
    def __repr__(self):
        """
        This returns a representation of a Sudoku object.

        :return: A string representing the Sudoku object.
        :rtype: str
        """
        # Change this to anything you like, such that you can easily print a Sudoku object.
        preview = ""
        self.sqrt = int(np.sqrt(self.size))
        
        for i in range(self.size):
            counter = 0
            for j in range(len(self.grid[i])):
                preview += str(int(self.grid[i][j])) + " "
                counter += 1
                if counter == self.sqrt and j != len(self.grid[i]) - 1:
                    preview += "| "
                    counter = 0
            preview = preview.rstrip()
            preview += "\n"
            if (i + 1) % self.sqrt == 0 and i != len(self.grid) - 1:
                preview += "-" * (2 * self.size + self.sqrt) + "\n"
        
        return preview


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
