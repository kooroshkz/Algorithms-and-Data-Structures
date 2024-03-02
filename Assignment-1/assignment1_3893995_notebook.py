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

############ CODE BLOCK 11 ################
    def set_grid(self, grid):
        """
        This method sets a new grid. This also can change the size of the sudoku.

        :param grid: A 2D numpy array that contains the digits for the grid.
        :type grid: ndarray[(Any, Any), int]
        """
        # simply set the object attribute grid to a new one with any value
        if len(grid) == len(self.grid):
            self.grid = grid
        else:
            raise ValueError("The size of the new grid does not match the size of the sudoku")

############ CODE BLOCK 12 ################
    def get_row(self, row_id):
        """
        This method returns the row with index row_id.

        :param row_id: The index of the row.
        :type row_id: int
        :return: A row of the sudoku.
        :rtype: np.ndarray[(Any,), int]
        """
        return self.grid[row_id, :]

    def get_col(self, col_id):
        """
        This method returns the column with index col_id.

        :param col_id: The index of the column.
        :type col_id: int
        :return: A row of the sudoku.
        :rtype: np.ndarray[(Any,), int]
        """
        return self.grid[:, col_id]

    def get_box_index(self, row, col):
        """
        This returns the box index of a cell given the row and column index.
        
        :param col: The column index.
        :type col: int
        :param row: The row index.
        :type row: int
        :return: This returns the box index of a cell.
        :rtype: int
        """
        if row > len(self.grid) or col > len(self.grid):
            raise ValueError("The row or column index is out of range.")
        
        index_size = int(len(self.grid)**0.5)
        col_box_id = 0
        row_box_id = 0
        # find the column box index
        for index in range(index_size):
            # determine in which box index column we are
            if index_size * (index + 1) > col and col >= index_size*index:
                col_box_id = index
            if index_size * (index + 1) > row and row >= index_size*index:
                row_box_id = index
        
        
        # create a numpy array with box indicies
        # for example for a 9x9 sudoku it will look like this:
        # [[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8]]
        index_array = np.arange(len(self.grid))
        index_array = index_array.reshape(index_size, index_size)
        
        # find the box index using coordinates found earlier
        return index_array[row_box_id][col_box_id]


    def get_box(self, box_id):
        """
        This method returns the "box_id" box.

        :param box_id: The index of the sudoku box.
        :type box_id: int
        :return: A box of the sudoku.
        :rtype: np.ndarray[(Any, Any), int]
        """
        index_size = int(len(self.grid)**0.5)
        index_array = np.arange(len(self.grid))
        index_array = index_array.reshape(index_size, index_size)
        for r_index, row in enumerate(index_array):
            for c_index, value in enumerate(row):
                if value == box_id:
                    return self.grid[r_index*index_size:(r_index+1)*index_size, c_index*index_size:(c_index+1)*index_size]


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
