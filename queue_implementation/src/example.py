from queue import Queue

def main():
    # Create a new queue
    queue = Queue()

    # Add some items
    print("Adding items to queue...")
    queue.enqueue("First")
    queue.enqueue("Second")
    queue.enqueue("Third")
    
    print(f"Queue size: {queue.size()}")
    print(f"Front item (peek): {queue.peek()}")

    # Remove and print items
    print("\nRemoving items from queue:")
    while not queue.is_empty():
        print(f"Dequeued: {queue.dequeue()}")

    print(f"\nQueue is empty: {queue.is_empty()}")

    # Demonstrate error handling
    try:
        queue.dequeue()
    except IndexError as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()