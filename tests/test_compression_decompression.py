import unittest
from hyperloglog.core import HyperLogLog, pack_registers, compress_sparse_registers

class TestCompression(unittest.TestCase):
    def test_dense_packing_roundtrip(self):
        hll = HyperLogLog(b=14, mode='dense')
        for i in range(100):
            hll.add(str(i))
        data = pack_registers(hll.registers, hll.b)
        self.assertIsInstance(data, bytes)

    def test_sparse_compression_roundtrip(self):
        hll = HyperLogLog(b=14, mode='sparse')
        for i in range(5):
            hll.add(str(i))
        data = compress_sparse_registers(hll.registers, hll.b)
        self.assertIsInstance(data, bytes)

if __name__ == '__main__':
    unittest.main(verbosity=2)

