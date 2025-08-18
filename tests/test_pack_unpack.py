import unittest
import random
from typing import List
from hyperloglog.compression import unpack_registers, pack_registers

import numpy as np
class TestRegisterPacking(unittest.TestCase):
    """Comprehensive unit tests for register packing functions."""

    def test_basic_functionality(self):
        """Test basic pack/unpack operations."""
        # Test empty case
        packed = pack_registers(np.array([]), 8)
        self.assertEqual(packed, b'')
        unpacked = unpack_registers(b'', 0, 8)
        self.assertTrue(np.array_equal(unpacked,  np.array([])))

        
        # Test single register
        packed = pack_registers(np.array([42]), 8)
        unpacked = unpack_registers(packed, 1, 8)
        self.assertEqual(unpacked, np.array([42]))
        
        # Test byte-aligned packing
        registers = np.array([1, 2, 3, 4, 5])
        packed = pack_registers(registers, 8)
        unpacked = unpack_registers(packed, len(registers), 8)
        self.assertTrue(np.array_equal(unpacked, registers))


    def test_sub_byte_packing(self):
        """Test bit packing for sub-byte register sizes."""
        # 4-bit registers
        registers = np.array([1, 2, 3, 4])
        packed = pack_registers(registers, 4)
        unpacked = unpack_registers(packed, len(registers), 4)
        self.assertTrue(np.array_equal(unpacked, registers))
        
        # 1-bit registers (bit array)
        registers = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        packed = pack_registers(registers, 1)
        unpacked = unpack_registers(packed, len(registers), 1)
        self.assertTrue(np.array_equal(unpacked, registers))


    def test_various_bit_widths_systematic(self):
        """Systematically test all bit widths from 1 to 16."""
        for binbits in range(1, 17):
            max_val = (1 << binbits) - 1
            
            # Test with various patterns
            test_patterns = [
                [0],
                [max_val],
                [0, max_val],
                [max_val, 0],
                list(range(min(8, max_val + 1))),  # Sequential values
            ]
            
            for pattern in test_patterns:
                pattern=np.array(pattern)
                if all(v <= max_val for v in pattern):
                    with self.subTest(binbits=binbits, pattern=pattern):
                        packed = pack_registers(pattern, binbits)
                        unpacked = unpack_registers(packed, len(pattern), binbits)
                        self.assertTrue(np.array_equal(unpacked, pattern))


    def test_stress_random_data(self):
        """Stress test with pseudo-random data."""
        random.seed(42)  # Reproducible
        
        for _ in range(100):
            binbits = random.randint(1, 16)
            max_val = (1 << binbits) - 1
            count = random.randint(1, 50)
            
            registers = np.array([random.randint(0, max_val) for _ in range(count)])
            
            packed = pack_registers(registers, binbits)
            unpacked = unpack_registers(packed, len(registers), binbits)
            self.assertTrue(np.array_equal(unpacked, registers))

    def test_pack_registers_value_errors(self):
        """Test pack_registers ValueError conditions with specific assertions."""
        # Test: registers must be a np.array
        with self.assertRaisesRegex(ValueError, "registers must be a numpy array"):
            pack_registers("not a list", 8)
        
        with self.assertRaisesRegex(ValueError, "registers must be a numpy array"):
            pack_registers((1, 2, 3), 8)  # tuple instead of list
        
        # Test: binbits must be positive integer
        with self.assertRaisesRegex(ValueError, "binbits must be a positive integer"):
            pack_registers(np.array([1]), 0)
        
        with self.assertRaisesRegex(ValueError, "binbits must be a positive integer"):
            pack_registers(np.array([1]), -1)
        
        with self.assertRaisesRegex(ValueError, "binbits must be a positive integer"):
            pack_registers(np.array([1]), 3.14)
        
        # Test: binbits must be <= 64
        with self.assertRaisesRegex(ValueError, "binbits must be <= 64 to prevent memory issues"):
            pack_registers(np.array([1]), 65)
        
        with self.assertRaisesRegex(ValueError, "binbits must be <= 64 to prevent memory issues"):
            pack_registers(np.array([1]), 100)
        
        # Test: Register values must be integers
        
        with self.assertRaisesRegex(ValueError, "Register 0 must be an integer"):
            pack_registers(np.array([3.14]), 8)
        
        # Test: Register values must be non-negative
        with self.assertRaisesRegex(ValueError, "Register 0 must be non-negative"):
            pack_registers(np.array([-1]), 8)
        
        with self.assertRaisesRegex(ValueError, "Register 2 must be non-negative"):
            pack_registers(np.array([1, 2, -5]), 8)
        
        # Test: Register values must fit in specified bit width
        with self.assertRaisesRegex(ValueError, "Register 0 value 256 exceeds 8-bit limit \\(255\\)"):
            pack_registers(np.array([256]), 8)
        
        with self.assertRaisesRegex(ValueError, "Register 1 value 16 exceeds 4-bit limit \\(15\\)"):
            pack_registers(np.array([1, 16]), 4)
        
        with self.assertRaisesRegex(ValueError, "Register 0 value 2 exceeds 1-bit limit \\(1\\)"):
            pack_registers(np.array([2]), 1)

    def test_unpack_registers_value_errors(self):
        """Test unpack_registers ValueError conditions with specific assertions."""
        # Test: data must be bytes
        with self.assertRaisesRegex(ValueError, "data must be bytes"):
            unpack_registers("not bytes", 1, 8)
        
        with self.assertRaisesRegex(ValueError, "data must be bytes"):
            unpack_registers([1, 2, 3], 1, 8)
        
        # Test: m must be non-negative integer
        with self.assertRaisesRegex(ValueError, "m must be a non-negative integer"):
            unpack_registers(b'\x01', -1, 8)
        
        with self.assertRaisesRegex(ValueError, "m must be a non-negative integer"):
            unpack_registers(b'\x01', 3.14, 8)
        
        # Test: binbits must be positive integer
        with self.assertRaisesRegex(ValueError, "binbits must be a positive integer"):
            unpack_registers(b'\x01', 1, 0)
        
        with self.assertRaisesRegex(ValueError, "binbits must be a positive integer"):
            unpack_registers(b'\x01', 1, -1)
        
        # Test: binbits must be <= 64
        with self.assertRaisesRegex(ValueError, "binbits must be <= 64 to prevent memory issues"):
            unpack_registers(b'\x01', 1, 65)
        
        # Test: Insufficient data
        with self.assertRaisesRegex(ValueError, "Insufficient data: need 2 bytes, got 1"):
            unpack_registers(b'\x01', 2, 8)  # Need 2 bytes for 2 8-bit registers
        
        with self.assertRaisesRegex(ValueError, "Insufficient data: need 2 bytes, got 1"):
            unpack_registers(b'\x01', 4, 4)  # Need 2 bytes for 4 4-bit registers
        
        with self.assertRaisesRegex(ValueError, "Insufficient data: need 3 bytes, got 1"):
            unpack_registers(b'\x01', 8, 3)  # Need 3 bytes for 8 3-bit registers

    def test_overflow_errors(self):
        """Test OverflowError conditions with specific assertions."""
        # Calculate limits
        limit = 2**20  # 1,048,576 bits
        exactly_at_limit = limit // 8  # 131,072 registers
        over_limit = exactly_at_limit + 1  # 131,073 registers
        
        # Test pack_registers overflow
        with self.assertRaisesRegex(OverflowError, "Total bits \\(1048584\\) too large, risk of memory overflow"):
            pack_registers(np.array([1] * over_limit), 8)
        
        # Test unpack_registers overflow - need enough data to pass the data check first
        large_data = b'\x01' * (over_limit + 1000)  # More than enough data
        with self.assertRaisesRegex(OverflowError, "Total bits \\(1048584\\) too large, risk of memory overflow"):
            unpack_registers(large_data, over_limit, 8)
        
        # Test that exactly at limit works (no overflow)
        try:
            packed = pack_registers(np.array([1] * exactly_at_limit), 8)
            unpacked = unpack_registers(packed, exactly_at_limit, 8)
            self.assertEqual(len(unpacked), exactly_at_limit)
        except (OverflowError, MemoryError):
            self.fail("Should not raise overflow at exactly the limit")

    def test_edge_cases(self):
        """Test various edge cases and boundary conditions."""
        # Test maximum values for different bit widths
        test_cases = [
            (1, [0, 1]),
            (2, [0, 1, 2, 3]),
            (4, [0, 7, 15, 8]),
            (8, [0, 127, 255, 128]),
        ]
        
        for binbits, registers in test_cases:
            registers=np.array(registers)
            with self.subTest(binbits=binbits):
                packed = pack_registers(registers, binbits)
                unpacked = unpack_registers(packed, len(registers), binbits)
                self.assertTrue(np.array_equal(unpacked, registers))

        
        # Test with extra data (should not cause issues)
        packed = pack_registers(np.array([1, 2]), 8)
        extra_data = packed + b'\xff\xff'
        unpacked = unpack_registers(extra_data, 2, 8)
        self.assertTrue(np.array_equal(unpacked, np.array([1,2])))

        
        # Test alternating patterns
        registers = np.array([0, 15, 0, 15, 0, 15])
        packed = pack_registers(registers, 4)
        unpacked = unpack_registers(packed, len(registers), 4)
        self.assertTrue(np.array_equal(unpacked, registers))

    def test_byte_calculation_correctness(self):
        """Test that byte calculations are correct for various bit arrangements."""
        test_cases = [
            (1, 1, 1),   # 1 bit -> 1 byte
            (1, 8, 1),   # 8 bits -> 1 byte  
            (1, 9, 2),   # 9 bits -> 2 bytes
            (3, 3, 2),   # 9 bits -> 2 bytes
            (5, 3, 2),   # 15 bits -> 2 bytes
            (6, 3, 3),   # 18 bits -> 3 bytes
        ]
        
        for register_count, binbits, expected_bytes in test_cases:
            with self.subTest(registers=register_count, binbits=binbits):
                registers = np.array([1] * register_count)
                packed = pack_registers(registers, binbits)
                self.assertEqual(len(packed), expected_bytes)
                
                unpacked = unpack_registers(packed, register_count, binbits)
                self.assertTrue(np.array_equal(unpacked, registers))



if __name__ == '__main__':
    unittest.main(verbosity=2)
