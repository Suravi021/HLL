'''to run without moving the hyerloglog folder use command
python -m unittest tests.test_compatibility'''
import unittest
from hyperloglog.core import HyperLogLog

class TestCompatibility(unittest.TestCase):
    def test_string_and_bytes(self):
        hll = HyperLogLog(b=14)
        hll.add("foo")
        hll.add(b"foo") 
        self.assertIsInstance(hll.estimate(), float)

if __name__ == "__main__":
    unittest.main(verbosity=2)
