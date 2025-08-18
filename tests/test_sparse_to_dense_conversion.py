import unittest
from hyperloglog.core import HyperLogLog

class TestSparseToDenseConversion(unittest.TestCase):
    def setUp(self):
        self.b = 14
        self.m = 1 << self.b

    def make_sparse(self, sparse_entries=None):
        hll = HyperLogLog(b=self.b, mode='sparse')
        if sparse_entries:
            hll.registers = sparse_entries.copy()
            hll.impl.registers = sparse_entries.copy()
        return hll

    def test_sparse_to_dense(self):
        many_entries = [(i, (i % 7) + 1) for i in range(int(self.m * 0.9))]
        hll1 = self.make_sparse(many_entries)
        hll2 = self.make_sparse([(1, 15), (3, 12), (7, 9)])

        hll1.merge(hll2)
        self.assertEqual(hll1.mode, 'dense')

if __name__ == "__main__":
    unittest.main()
