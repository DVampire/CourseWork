import unittest
from src.queue import Queue

class TestQueue(unittest.TestCase):
    def setUp(self):
        self.queue = Queue()

    def test_empty_queue(self):
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)

    def test_enqueue(self):
        self.queue.enqueue(1)
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 1)
        self.assertEqual(self.queue.front(), 1)

    def test_dequeue(self):
        self.queue.enqueue(1)
        self.queue.enqueue(2)
        self.assertEqual(self.queue.dequeue(), 1)
        self.assertEqual(self.queue.size(), 1)
        self.assertEqual(self.queue.dequeue(), 2)
        self.assertTrue(self.queue.is_empty())

    def test_front(self):
        self.queue.enqueue("first")
        self.queue.enqueue("second")
        self.assertEqual(self.queue.front(), "first")
        self.assertEqual(self.queue.size(), 2)

    def test_clear(self):
        self.queue.enqueue(1)
        self.queue.enqueue(2)
        self.queue.clear()
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.size(), 0)

    def test_empty_queue_operations(self):
        with self.assertRaises(IndexError):
            self.queue.dequeue()
        with self.assertRaises(IndexError):
            self.queue.front()

if __name__ == '__main__':
    unittest.main()