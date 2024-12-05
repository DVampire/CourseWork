class Queue:
    def __init__(self):
        """Initialize an empty queue."""
        self._items = []

    def enqueue(self, item):
        """Add an item to the back of the queue.
        
        Args:
            item: The item to be added to the queue
        """
        self._items.append(item)

    def dequeue(self):
        """Remove and return the item at the front of the queue.
        
        Returns:
            The item at the front of the queue
            
        Raises:
            IndexError: If the queue is empty
        """
        if self.is_empty():
            raise IndexError("Cannot dequeue from an empty queue")
        return self._items.pop(0)

    def front(self):
        """Return the item at the front of the queue without removing it.
        
        Returns:
            The item at the front of the queue
            
        Raises:
            IndexError: If the queue is empty
        """
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]

    def is_empty(self):
        """Check if the queue is empty.
        
        Returns:
            bool: True if the queue is empty, False otherwise
        """
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the queue.
        
        Returns:
            int: The number of items in the queue
        """
        return len(self._items)

    def clear(self):
        """Remove all items from the queue."""
        self._items = []