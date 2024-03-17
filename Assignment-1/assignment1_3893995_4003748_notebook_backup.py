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
        preview = "" # Empty string to add the sudoku grid to.

        # To avoid multiple attribute lookups
        max_digits = len(str(self.size))
        sqrt = int(self.size ** 0.5)
        grid = self.grid  
        
        # Iterate through each row of the grid.
        for row in range(self.size):
            # Iterate through each element in the row.
            for column in range(len(grid[row])):
                cell_value = int(grid[row][column])
                digits = len(str(cell_value)) if cell_value != 0 else 1
                preview += str(cell_value) + " " * (max_digits - digits + 1)
                
                # Add a vertical separator after every square root number of elements, except for the last element in the row.
                if (column + 1) % sqrt == 0 and column != len(grid[row]) - 1:
                    preview += " "
            
            # Remove trailing whitespace and add a newline after every square root number of rows.
            preview = preview.rstrip()
            preview += "\n"
            
            # Add a horizontal separator after every square root number of rows, except for the last row.
            if (row + 1) % sqrt == 0 and row != len(grid) - 1:
                preview += "\n"
        
        return preview

############ CODE BLOCK 11 ################
    def set_grid(self, grid):
        """
        This method sets a new grid. This also can change the size of the sudoku.

        :param grid: A 2D numpy array that contains the digits for the grid.
        :type grid: ndarray[(Any, Any), int]
        """
        # simply set the object attribute grid to a new one with any value if it matches the size
        if len(grid) == len(self.grid):
            self.grid = grid
        else:
            raise ValueError("The size of the new grid does not match the size of the sudoku")


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
