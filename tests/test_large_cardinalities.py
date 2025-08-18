import unittest
from hyperloglog.core import HyperLogLog

class TestLargeCardinalities(unittest.TestCase):
    def test_large_cardinality(self):
        hll = HyperLogLog(b=14)
        for i in range(120_000):
            hll.add(str(i))
        est = hll.estimate()
        self.assertGreater(est, 100_000)
        self.assertLess(abs(est - 120_000)/120_000, 0.05)  # <5% error

if __name__ == '__main__':
    unittest.main(verbosity=2)