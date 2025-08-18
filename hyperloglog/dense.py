from bisect import bisect_left
import math
import numpy as np

from .constants import ALPHA_MM, THRESHOLD
from .bias_correction import bias_estimate
from .hash_utils import murmurhash64a
from .compression import unpack_registers

class DenseHyperLogLog:
    """
    Dense HyperLogLog implementation .
    """
    def __init__(self, b: int = 14, register: bytes = b''):
        """
        Initializes the DenseHyperLogLog instance.

        Args:
            b (int): Precision parameter (number of bits for indexing registers). Default is 14.
            register (bytes): Packed register state or 0 for a fresh instance.

        Notes:
            - The number of registers m is 2^b.
            - If `register` is provided, it is unpacked into register values.
        """
        if b in range(4, 19):
            self.b = b # number of bits in the register
        else:
             raise ValueError("Value of b not in range [4,18]")
        
        self.m = 1 << b

        if register:
            self.registers = unpack_registers(register, 1 << self.b, self.b)

        else:
            self.registers = np.zeros(self.m, dtype=np.int32)

            
    def add(self, item: str) -> int:
        """
        Adds a single item to the HLL estimator.

        Args:
            item (str): The item to add (already stringified externally).

        Returns:
            int: Always returns 0 (placeholder for compatibility with sparse mode).
        """
        
        hash_value = murmurhash64a(item)
        idx = hash_value >> (64 - self.b)
        w = (hash_value << self.b) & ((1 << 64) - 1)
        rho = self._rho(w, 64 - self.b)
        self.registers[idx] = max(self.registers[idx], rho)
        return 0

    def _rho(self, w: int, max_bits: int) -> int:
        """
        Computes the position of the first set bit (rho value) in the hash, adjusted for noise.

        Args:
            w (int): The hashed value.
            max_bits (int): The maximum number of bits allowed.

        Returns:
            int: The rho value, capped at `max_bits`.

        Notes:
            - Implements a fallback re-hashing loop if rho is suspiciously large.
        """
        def clzll(x): return 64 - x.bit_length() if x != 0 else 64
        rho = clzll(w) + 1
        
        if rho >= 64:
            max_val = min(1 << 6, max_bits)  # cap with max_bits
            safety_counter = 0               
            while rho < max_val:
                w = murmurhash64a(str(w))
                addn = clzll(w) + 1
                if addn <= 0:
                    break  # avoid infinite loop
                rho += addn
                safety_counter += 1
                if safety_counter > max_bits:  # hard safety cutoff
                    break

        return min(rho, max_bits)

    def estimate(self) -> float:
        """
        Estimates the cardinality of the current multiset based on register values.

        Returns:
            float: The estimated number of unique elements.

        Notes:
            - Applies raw HyperLogLog formula for large cardinalities.
            - Uses bias correction for mid-range estimates.
            - Uses linear counting for small cardinalities with many zero registers.
        """
        m = self.m
        Z = sum(2.0 ** -r for r in self.registers)
        E = ALPHA_MM[self.b] / Z
        V = np.count_nonzero(self.registers == 0)

        if E <= THRESHOLD[self.b]:
            correction = bias_estimate(E, self.b)
            E -= correction
            if E < 0:
                E = 0
        if V > 0:
            H = m * math.log(m / V)
            if H <= THRESHOLD[self.b]:
                return H
        return E
