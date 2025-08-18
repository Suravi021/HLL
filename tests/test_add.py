
from hyperloglog.core import HyperLogLog
from hyperloglog.sparse import SparseHyperLogLog
import unittest
from hyperloglog.dense import DenseHyperLogLog


class TestCoreAdd(unittest.TestCase):

    def test_add_stays_sparse_below_threshold(self):
        hll = HyperLogLog(b=14)  # starts sparse threshhold= m//4 for 14 it is 4096
        sparse_threshold = hll.impl.sparse_threshold

        for i in range(sparse_threshold):
            hll.add(f"item-{i}")

        self.assertIsInstance(hll.impl, SparseHyperLogLog)

    def test_add_converts_to_dense_above_threshold(self):
        hll = HyperLogLog(b=14)  # starts sparse
        sparse_threshold = hll.impl.sparse_threshold

        for i in range(sparse_threshold + 1002):  # random more than threshold
            hll.add(f"item-{i}")

        self.assertIsInstance(hll.impl, DenseHyperLogLog)


if __name__ == '__main__':
    unittest.main(verbosity=2)


