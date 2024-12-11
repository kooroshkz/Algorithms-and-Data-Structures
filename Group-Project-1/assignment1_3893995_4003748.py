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
        if int(grid.shape[0]**0.5) ** 2 == grid.shape[0]:
            self.size = grid.shape[0]
            self.grid = grid
        else:
            raise ValueError("The size of the new grid does not match an standard sudoku")

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
                    # slice the sudoku at respective box indicies
                    return self.grid[r_index*index_size:(r_index+1)*index_size, c_index*index_size:(c_index+1)*index_size]

############ CODE BLOCK 13 ################
    @staticmethod
    def is_set_correct(numbers):
        """
        This method checks if a set (row, column, or box) is correct according to the rules of a sudoku.
        In other words, this method checks if a set of numbers contains duplicate values between 1 and the size of the sudoku.
        Note, that multiple empty cells are not considered duplicates.

        :param numbers: The numbers of a sudoku's row, column, or box.
        :type numbers: np.ndarray[(Any, Any), int] or np.ndarray[(Any, ), int]
        :return: This method returns if the set is correct or not.
        :rtype: Boolean
        """

        try:
            if numbers.shape[1]:
                numbers = numbers.flatten()
        except IndexError:
            pass
        
        # filter out zeros
        filtered_numbers = [number for number in numbers if number != 0]
        
        # check if the filtered set contains duplicates
        return len(filtered_numbers) == len(set(filtered_numbers))


    def check_cell(self, row, col):
        """
        This method checks if the cell, denoted by row and column, is correct according to the rules of sudoku.
        
        :param col: The column index that is tested.
        :type col: int
        :param row: The row index that is tested.
        :type row: int
        :return: This method returns if the cell, denoted by row and column, is correct compared to the rest of the grid.
        :rtype: boolean
        """
        # check if the row, column and box are correct
        is_row_correct = self.is_set_correct(self.get_row(row))
        is_col_correct = self.is_set_correct(self.get_col(col))
        is_box_correct = self.is_set_correct(self.get_box(self.get_box_index(row, col)))
        
        return is_row_correct and is_col_correct and is_box_correct

    def check_sudoku(self):
        """
        This method checks, for all rows, columns, and boxes, if they are correct according to the rules of a sudoku.
        In other words, this method checks, for all rows, columns, and boxes, if a set of numbers contains duplicate values between 1 and the size of the sudoku.
        Note, that multiple empty cells are not considered duplicates.

        Hint: It is not needed to check if every cell is correct to check if a complete sudoku is correct.

        :return: This method returns if the (partial) Sudoku is correct.
        :rtype: Boolean
        """

        # check if all rows, columns and boxes are correct to show if the sudoku is correct.
        for index in range(len(self.grid)):
            if not self.is_set_correct(self.get_row(index)) or not self.is_set_correct(self.get_col(index)) or not self.is_set_correct(self.get_box(index)):
                return False
        return True

############ CODE BLOCK 14 ################
    def step(self, row=0, col=0, backtracking=False):
        """
        This is a recursive method that completes one step in the exhaustive search algorithm.
        A step should contain at least, filling in one number in the sudoku and calling "next_step" to go to the next step.
        If the current number for this step does not give a correct solution another number should be tried 
        and if no numbers work the previous step should be adjusted.

        This method should work for both backtracking and exhaustive search.
        
        Hint 1: Numbers, that are already filled in should not be overwritten.
        Hint 2: Think about a base case.
        Hint 3: The step method from the previous jupyter notebook cell can be copy-paste here and adjusted for backtracking.
    
        :param col: The current column index.
        :type col: int
        :param row: The current row index.
        :type row: int
        :param backtracking: This determines if backtracking is used, defaults to False.
        :type backtracking: boolean, optional
        :return: This method returns if a correct solution can be found using this step.
        :rtype: boolean
        """

        # If the current cell is already filled, move to the next
        if self.grid[row][col] != 0:
            return self.next_step(row, col, backtracking)

       
        # Iterate all possible numbers
        for number in range(1, len(self.grid) + 1):
            # Fill in the number
            self.grid[row][col] = number

            # determine if we use backtracking and if yes,
            # then check if we can add the current cell,
            # if not, then try the next number
            if backtracking and not self.check_cell(row, col):
                continue

            # Move to the next one
            if self.next_step(row, col, backtracking):
                return True
            
        self.clean_up(row, col)
        return False


    def next_step(self, row, col, backtracking):
        """
        This method calculates the next step in the recursive exhaustive search algorithm.
        This method should only determine which cell should be filled in next.

        This method should work for both backtracking and exhaustive search.
        
        :param col: The current column index.
        :type col: int
        :param row: The current row index.
        :type row: int
        :param backtracking: This determines if backtracking is used, defaults to False.
        :type backtracking: boolean, optional
        :return: This method returns if a correct solution can be found using this next step.
        :rtype: boolean
        """

        # reach the end and check if the solution is correct
        # this is done here and not in step, because we need to add the 
        # value corresponding to the last cell before returning the sudoku
        if row == len(self.grid) - 1 and col == len(self.grid) - 1:
            return self.check_sudoku()

        # calculate the next row index using integer division, we only increment the row
        # when we are at the last column
        next_row = row + (col + 1) // len(self.grid)  
        # by using modulo arithmetic we can easily get the next column, as long as 
        # we are not at the right end of the grid we keep increasing the column, if we reach 
        # the end the column is then reset to 0
        next_col = (col + 1) % len(self.grid) 
        return self.step(next_row, next_col, backtracking)
    
    def clean_up(self, row, col):
        """
        This method cleans up the current cell if no solution can be found for this cell.

        This method should work for both backtracking and exhaustive search.
        
        :param col: The current column index.
        :type col: int
        :param row: The current row index.
        :type row: int
        :return: This method returns if a correct solution can be found using this next step.
        :rtype: boolean
        """
        # clean up the cell
        self.grid[row][col] = 0
    
    def solve(self, backtracking=False):
        """
        Solve the sudoku using recursive exhaustive search or backtracking.
        This is done by calling the "step" method, which does one recursive step.
        This can be visualized as a process tree, where "step" completes the functionality of of node.
        
        This method is already implemented and you do not have to do anything here.

        :param backtracking: This determines if backtracking is used, defaults to False.
        :type backtracking: boolean, optional
        :return: This method returns if a correct solution for the whole sudoku was found.
        :rtype: boolean
        """
        return self.step(backtracking=backtracking)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
