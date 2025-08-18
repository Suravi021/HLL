import unittest
from hyperloglog.core import HyperLogLog

class TestInvalidParameters(unittest.TestCase):
    def test_invalid_b(self):
        with self.assertRaises(ValueError):
            HyperLogLog(b=2)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            HyperLogLog(b=14, mode='nonsense')

    def test_merge_with_none(self):
        with self.assertRaises(TypeError):
            HyperLogLog.merge(None)

    def test_merge_with_string(self):
        with self.assertRaises(TypeError):
            HyperLogLog.merge("invalid")

    def test_add_none(self):
        with self.assertRaises(TypeError):
            HyperLogLog.add(None)

if __name__ == '__main__':
    unittest.main(verbosity=2)
