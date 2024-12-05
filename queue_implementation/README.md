# Queue Implementation

A Python implementation of a Queue data structure for coursework assignment.

## Features

- Basic queue operations (enqueue, dequeue)
- Front element access
- Queue size and empty status checks
- Clear operation
- Comprehensive unit tests
- Example usage script

## Project Structure

```
queue_implementation/
├── src/
│   └── queue.py       # Main queue implementation
├── tests/
│   └── test_queue.py  # Unit tests
└── example.py         # Usage example
```

## Usage

```python
from src.queue import Queue

# Create a new queue
queue = Queue()

# Add items
queue.enqueue("First")
queue.enqueue("Second")

# Get the front item
front_item = queue.front()  # Returns "First"

# Remove items
first = queue.dequeue()  # Returns "First"
second = queue.dequeue()  # Returns "Second"

# Check if empty
is_empty = queue.is_empty()  # Returns True
```

## Running Tests

```bash
python -m unittest tests/test_queue.py -v
```

## Running Example

```bash
python example.py
```