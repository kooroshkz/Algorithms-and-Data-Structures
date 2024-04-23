import os
import re
import shutil
import numpy as np
from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, Magics, magics_class

END_OF_CODE_SIGN = "\n############ END OF CODE BLOCKS, START SCRIPT BELOW! ################\n"

@magics_class
class CustomMagics(Magics):
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-a', '--append', action='store_true', default=False,
        help='Append contents of the cell to an existing file. '
             'The file will be created if it does not exist.'
    )
    @magic_arguments.argument(
        '-c', '--skip_class', action='store_true', default=False,
        help='Skip all lines of code including the line that contains the key word class.'
             'This makes it possible to continue to add functionality to a class.'
    )
    @magic_arguments.argument(
        '-s', '--sorting', action='store_true', default=False,
        help='If this flag is enabled all code blocks will be sorted and reordered'
    )
    @magic_arguments.argument(
        'filename', type=str,
        help='file to write'
    )
    @magic_arguments.argument(
        'code_block', type=int,
        help='file to write'
    )
    @cell_magic
    def execwritefile(self, line, cell):
        """Write the contents of the cell to a file.

        The file will be overwritten unless the -a (--append) flag is specified.
        """
        args = magic_arguments.parse_argstring(self.execwritefile, line)

        # Remove quotes from the filename
        if re.match(r'^(\'.*\')|(".*")$', args.filename):
            filename = os.path.expanduser(args.filename[1:-1])
        else:
            filename = os.path.expanduser(args.filename)

        # Make a backup if the file is overwritten
        exist_flag = os.path.exists(filename)
        if exist_flag and not args.append:
            print(f"{filename} is backup to {filename[:-3]}_backup.py")
            shutil.copy(filename, f"{filename[:-3]}_backup.py")

        # Handle how to write to file and print message
        mode = 'r+' if args.append else 'w'

        # Clean up all scripts (This needs a seperate read/write)
        if exist_flag and args.append:
            CustomMagics.making_code_block_ending(filename, clean_up=True)

        replace_flag = False
        with open(filename, mode, encoding='utf-8') as f:
            # Check if the code already exist in the file and overwrite that part
            if exist_flag and args.append and f"CODE BLOCK {args.code_block}" in (code := f.read()):
                print(f"Replace existing code {filename}")
                replace_flag = True
            else:
                # Check if file exist to overwrite or append
                if exist_flag:
                    if args.append:
                        print(f"Appending to {filename}")
                    else:
                        print(f"Overwriting {filename}")
                else:  # Make new file
                    print(f"Writing {filename}")

                # Write content
                CustomMagics.write_code_to_file(f, cell, args.code_block, args.skip_class)

        # The file must be read again
        if replace_flag:
            CustomMagics.replace_code(filename, code, cell, args)

        # Correct the file ending
        CustomMagics.making_code_block_ending(filename)

        # The updated file is read again and sorted
        if args.sorting:
            CustomMagics.reorder_code(filename)

        # Correct the file ending
        CustomMagics.making_code_block_ending(filename)

        # Execute the cell content in the jupyter notebook
        get_ipython().run_cell(cell)

    @staticmethod
    def making_code_block_ending(filename, clean_up=False):
        with open(filename, "r", encoding='utf-8') as f:
            code = f.read()

        index = code.find(END_OF_CODE_SIGN)
        if index == -1:
            new_code = code + END_OF_CODE_SIGN
        elif clean_up:
            new_code = code[:index] + END_OF_CODE_SIGN
        else:
            new_code = code[:index] + code[index+len(END_OF_CODE_SIGN):] + END_OF_CODE_SIGN

        # write the content in the new order
        with open(filename, "w", encoding='utf-8') as f:
            f.write(new_code)

    @staticmethod
    def write_code_to_file(f, cell, code_block, skip_class):
        f.write(f"############ CODE BLOCK {code_block} ################\n")
        # Searches for the first mentioned of class and skips all code including the line containing the class definition.
        if skip_class:
            # Remove all lines not containing "class" till you find the line starting with class
            while (index := cell.find("class")) != 0:
                # Check if the code block contains a class.
                if index == -1:
                    raise ValueError("This code block does not contain a class. Either remove the -c flag or add a class.")

                # skip to the next line. This can skip the word class if it is not used as a key word.
                cell = cell[cell.find("\n")+1:]

            # skip one more line after you find the key word class.
            cell = cell[cell.find("\n")+1:]

        cell = cell.rstrip()
        f.write(cell)
        f.write("\n\n")

    @staticmethod
    def reorder_code(filename):
        # read the current (updated content)
        with open(filename, "r", encoding='utf-8') as f:
            code = f.read()

        # find code blocks and their indentifier
        code_block_indices = [m.span() for m in re.finditer("############ CODE BLOCK (\d+) ################", code)]
        code_block_id = [int(re.search("\d+", code[start:end]).group()) for start, end in code_block_indices]
        # add end of file.
        code_block_indices.append((code.find(END_OF_CODE_SIGN)+len(END_OF_CODE_SIGN),))

        # sort the code blocks
        new_code = ""
        for id in np.argsort(code_block_id):
            start = code_block_indices[id][0]
            end = code_block_indices[id+1][0]
            new_code += code[start:end]

        # write the content in the new order
        with open(filename, "w", encoding='utf-8') as f:
            f.write(new_code)

    @staticmethod
    def replace_code(filename, code, cell, args):
        """
        This function finds the position of the code block that needs to replaced and
        overwrite the old file with the code before the new code block, the new code block, and
        the code behind the old code block.
        """
        # Find code blocks
        new_code_block_pos = code.find(f"############ CODE BLOCK {args.code_block} ################")
        next_code_block_pos = code.find(f"############ CODE BLOCK", new_code_block_pos+40)

        # Rewrite the file
        with open(filename, "w", encoding='utf-8') as f:
            f.write(code[:new_code_block_pos])
            CustomMagics.write_code_to_file(f, cell, args.code_block, args.skip_class)
            if next_code_block_pos != -1:
                f.write(code[next_code_block_pos:])
