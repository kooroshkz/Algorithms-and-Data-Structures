############ CODE BLOCK 0 ################
# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

RNG = np.random.default_rng()

############ CODE BLOCK 10 ################
class Link():
    """
    This class creates a node that can be used to build any kind of linked list.
    Note, that we called it Link to distinguish this class from the DLink class
    which is used as nodes for double-linked lists.
    
    Attributes:
        :param self.info: The value of the node.
        :type self.info: int
        :param self.next: The next node in the linked list, defaults to None.
        :type self.next: List, optional 
    """
    def __init__(self, info, next=None):
        self.info = info
        self.next = next

    def __repr__(self):
        """
        This returns a representation of a Link object.

        :return: A string representing the Link object.
        :rtype: str
        """
        # Change this to anything you like, such that you can easily print a Node object.
        return super(Link, self).__repr__() 

class DLink(Link):
    """
    This class creates a node that can be used to build any kind of double-linked list.
    Note, that we called it DLink to distinguish this class from the Link class
    which is used as nodes for linked lists.
    
    Attributes:
        :param self.info: The value of the node. (inherited)
        :type self.info: int
        :param self.next: The next node in the double-linked list, defaults to None. (inherited)
        :type self.next: DList, optional 
        :param self.prev: The previous node in the double-linked list, defaults to None.
        :type self.prev: DList, optional         
    """
    def __init__(self, info, previous=None, next=None):
        super(DLink, self).__init__(info, next)  # Inherrit the init method of Link. You can think of this as copy-pasting the link init here.
        self.prev = previous

############ CODE BLOCK 20 ################
class BasicLinkedList():
    """
    This class creates linked list objects.
    The most basic implementation of a linked list stores 
    only the start of the list and nothing else.
    This is what we will use during this exercise.
    
    This means that you can only access the start node directly, 
    while other nodes can be accessed through the start node. 

    Attributes:
        :param self.start: The start node of the single-linked list, defaults to None.
        :type self.start: Link, optional
    """
    def __init__(self, start=None):
        """
        This initializes a single linked list object.
        Note, that this creates by default an empty list.
        """
        self.start = start
        
    def __repr__(self):
        """
        This returns a representation of a LinkedList object.

        :return: A string representing the LinkedList object.
        :rtype: str
        """
        if self.start is None:
            return "[]"
        else:
            result = "["
            current = self.start
            while current is not None:
                result += f"{current.info}, "
                current = current.next
            result = result.rstrip(", ") + "]"
            return result

class BasicDLinkedList(BasicLinkedList):
    """
    This class creates double-linked list objects.
    A double-linked list stores both the start and end of a list.
    This is what we will use during this exercise.
    
    This means that you can only access the start and end node directly, 
    while other nodes can be accessed through either the start node or the end node. 

    Note, that this class uses DLink objects and note Link objects.
    
    Attributes:
        :param self.start: The start node of the double-linked list, defaults to None.
        :type self.start: DLink, optional
        :param self.end: The end node of the double-linked list, defaults to None.
        :type self.end: DLink, optional
    """
    
    def __init__(self, start=None, end=None):
        """
        This initializes a basic double-linked list object.
        Note, that this creates by default an empty list.
        """
        super(BasicDLinkedList, self).__init__(start)  # Inherrit the init method of Link. You can think of this as copy-pasting the link init here.
        if end is None:
            self.end = start
        else:
            self.end = end

############ CODE BLOCK 21 ################
class QueueSingle(BasicLinkedList):
    """
    This class has the same attributes, initialization, and representation as the BasicLinkedList.
    """
    def pop(self):
        """
        This method removes the first element in the queue and returns it.
        
        If the list is empty raise the following error "IndexError: pop from empty list."
        You can do this with "raise IndexError(message)"

        :return: This returns the info of the first link in the linked list.
        :rtype: int
        """
        if self.start is None:
            raise IndexError("pop from empty list")
        else:
            first_info = self.start.info
            self.start = self.start.next
            return first_info

    def append(self, value):
        """
        This method adds a new element to the queue.
        In a queue, an element is always placed at the end of the linked list.

        :param value: This is the value that needs to be added to the linked list.
        :type value: int
        """
        new_link = Link(value)
        if self.start is None:
            self.start = new_link
        else:
            current = self.start
            while current.next is not None:
                current = current.next
            current.next = new_link         

############ CODE BLOCK 30 ################
class Queue(BasicDLinkedList):
    """
    This class has the same attributes, initialization, and representation as the BasicDLinkedList.
    """
    def pop(self):
        """
        This method removes the first element in the queue and returns it.
        
        If the list is empty raise the following error "IndexError: pop from empty list."
        You can do this with "raise IndexError(message)"

        :return: This returns the info of the first link in the linked list.
        :rtype: int
        """
        if self.start is None:
            raise IndexError("pop from empty list")
        else:
            first_info = self.start.info
            self.start = self.start.next
            if self.start is None:
                self.end = None
            return first_info

    def append(self, value):
        """
        This method adds a new element to the queue.
        In a queue, an element is always placed at the end of the linked list.

        :param value: This is the value that needs to be added to the queue.
        :type value: int
        """
        new_link = DLink(value, previous=self.end)
        if self.start is None:
            self.start = new_link
            self.end = new_link
        else:
            self.end.next = new_link
            self.end = new_link

############ CODE BLOCK 40 ################
class SingleLinkedList(BasicLinkedList):
    def pop(self, n):
        """
        This method removes the nth element in the list and returns it.
        
        If the list is empty raise the following error "IndexError: pop from empty list".
        You can do this with "raise IndexError(message)".

        If the value of n is greater than or equal to the length of the list 
        raise the following error "IndexError: pop index out of range".
        Again, You can do this with "raise IndexError(message)".

        :param n: This value determines which Link info is returned and which Link needs to be removed.
        :type n: int
        :return: This returns the info of the nth link in the linked list.
        :rtype: int
        """
        if self.start is None:
            raise IndexError("pop from empty list")
        
        if n < 0:
            raise IndexError("pop index out of range")
        
        if n == 0:
            return_value = self.start.info
            self.start = self.start.next
            return return_value
        
        current = self.start
        for i in range(n - 1):
            if current.next is None:
                raise IndexError("pop index out of range")
            current = current.next
        
        return_value = current.next.info
        current.next = current.next.next
        return return_value
        
    def insert(self, value, n):
        """
        This method inserts at the nth element in the list a new value.
        This means that all other values are essentially pushed one index ahead. 
        This is a consequence of inserting a value in the "middle" of the list.
     
        If the value of n is greater than the length of the list,
        you need to insert the value at the end of the list.

        :param value: This is the value that needs to be added to the linked list.
        :type value: int
        :param n: This value determines where the new value needs to be inserted.
        :type n: int
        """
        new_link = Link(value)
        if self.start is None:
            self.start = new_link
            return
        
        if n <= 0:
            new_link.next = self.start
            self.start = new_link
            return
        
        current = self.start
        for _ in range(n - 1):
            if current.next is None:
                break
            current = current.next
        
        new_link.next = current.next
        current.next = new_link

############ CODE BLOCK 41 ################
    def search(self, value):
        """
        This method is a basic search method to check if a value is in the list.
        If the value is in the list return True otherwise return False.

        :param value: This is the value that is searched for.
        :type value: int
        :return: A boolean is returned containing the answer if the search was successful.
        :rtype: boolean
        """
        current = self.start
        while current is not None:
            if current.info == value:
                return True
            current = current.next
        return False

############ CODE BLOCK 45 ################
class DoubleLinkedList(BasicDLinkedList):
    def pop(self, n):
        """
        This method removes the nth element in the list and returns it.
        
        If the list is empty raise the following error "IndexError: pop from empty list".
        You can do this with "raise IndexError(message)".

        If the value of n is greater than or equal to the length of the list 
        raise the following error "IndexError: pop index out of range".
        Again, You can do this with "raise IndexError(message)".

        :param n: This value determines which DLink info is returned and which DLink needs to be removed.
        :type n: int
        :return: This returns the info of the nth link in the double-linked list.
        :rtype: int
        """
        if self.start is None:
            raise IndexError("pop from empty list")
        
        if n < 0:
            raise IndexError("pop index out of range")
        
        if n == 0:
            return_value = self.start.info
            self.start = self.start.next
            if self.start is not None:
                self.start.prev = None
            return return_value
        
        current = self.start
        for i in range(n):
            if current.next is None:
                raise IndexError("pop index out of range")
            current = current.next
        
        return_value = current.info
        current.prev.next = current.next
        if current.next is not None:
            current.next.prev = current.prev
        return return_value
           
    def insert(self, value, n):
        """
        This method inserts at the nth element in the list a new value.
        This means that all other values are essentially pushed one index ahead. 
        This is a consequence of inserting a value in the "middle" of the list.
     
        If the value of n is greater than the length of the list,
        you need to insert the value at the end of the list.

        :param value: This is the value that needs to be added to the double-linked list.
        :type value: int
        :param n: This value determines where the new value needs to be inserted.
        :type n: int
        """
        new_link = DLink(value)
        if self.start is None:
            self.start = new_link
            return
        
        if n <= 0:
            new_link.next = self.start
            self.start.prev = new_link
            self.start = new_link
            return
        
        current = self.start
        for _ in range(n - 1):
            if current.next is None:
                break
            current = current.next
        
        new_link.prev = current
        new_link.next = current.next
        if current.next is not None:
            current.next.prev = new_link
        current.next = new_link

############ CODE BLOCK 46 ################
    def search(self, value):
        """
        This method is a basic search method to check if a value is in the list.
        If the value is in the list return True otherwise return False.

        :param value: This is the value that is searched for.
        :type value: int
        :return: A boolean is returned containing the answer if the search was successful.
        :rtype: boolean
        """
        current = self.start
        while current is not None:
            if current.info == value:
                return True
            current = current.next
        return False


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################