from .dense import DenseHyperLogLog
from .sparse import SparseHyperLogLog
from .compression import pack_registers, compress_sparse_registers
import base64
import struct

class HyperLogLog:
    """
    HyperLogLog (HLL) main interface for cardinality estimation.
    Supports both dense and sparse representations.
    """
    def __init__(self, b: int = 14, mode: str = 'sparse', register: int | bytes = 0):
        """
        Initializes the HyperLogLog object.

        Args:
            b: int - number of bits used to index registers (precision parameter)
            mode: str - 'dense' or 'sparse' mode
            register: int/bytes - initial register data (0 for empty)
        """
        if b in range(4,19):
            self.b = b # number of bits in the register
        else:
             raise ValueError("Value of b not in range [4,18]")
            
        self.mode = mode # dense or sparse 
        self.m = 1 << b
        mode = mode.lower()
        if mode == 'dense':
            self.impl = DenseHyperLogLog(b, register)
        elif mode == 'sparse':
            self.impl = SparseHyperLogLog(b, register)           
        else:
            raise ValueError('Unknown mode: ' + str(mode) + '.mode either be sparse or dense')
        self.registers = self.impl.registers

    def add(self, item: object ) -> None :
        """
            Adds an item to the HyperLogLog estimator, managing sparse/dense mode automatically.

            Converts the item to a string, hashes it, and updates the current
            implementation (sparse or dense). If sparse mode exceeds its threshold,
            the HLL is promoted to dense mode.

            Args:
                item (object): Any hashable object to add.

            Notes:
                - Mode switching is automatic; callers do not manage it.
                - The cardinality estimate is approximate and probabilistic.
                - This method mutates the HLL state and returns nothing.
        """
                
        if (self.impl.add(str(item))):
            print("Converting to Dense")
            self.convert_to_dense()

    def estimate(self) -> float:
        """
        Returns:
            float: estimated cardinality based on the current HLL state.
        """
        return self.impl.estimate()

    def storing(self) -> bytes:
        """
        Serializes the HLL registers for storage or transmission.

        Returns:
            bytes: packed register data
        """
        if self.mode == 'dense':
            return pack_registers(self.registers, self.b)
        else:
            return compress_sparse_registers(self.registers, self.b )

        
    def convert_to_dense(self):
        """
        Converts the current HLL from sparse to dense mode.
        Updates internal implementation and registers.
        """

        # Build dense registers from sparse data
        if self.mode == 'sparse':
            if isinstance(self.impl, SparseHyperLogLog):
                sparse_regs = self.impl.registers
                dense_registers = [0] * self.m
                for idx, rho in sparse_regs:
                    dense_registers[idx] = rho

                packed = pack_registers(dense_registers, self.b)

                self.mode = 'dense'
                self.impl = DenseHyperLogLog(self.b, packed)
                self.registers = self.impl.registers


    def merge(self, hll2):
            """
            Merges another HLL object into this one, matching C implementation logic.
        
            Args:
                hll2: HyperLogLog - another HLL with the same precision parameter b
        
            Returns:
                HLL object: the currect HLL object with hll2's registers.
        
            Raises:
                ValueError: if the precision parameter (b) values do not match
            """
            
            if self.b != hll2.b:
                raise ValueError("Cannot merge HLLs with different precision (b) values")

            m = 1 << self.b

            if self.mode == 'dense' and hll2.mode == 'dense':
                self.impl.registers = [
                    max(self.impl.registers[i], hll2.impl.registers[i]) for i in range(m)
                ]
                return self

            if self.mode == 'dense' and hll2.mode == 'sparse':
                for idx, rho in hll2.registers:
                    if rho > self.impl.registers[idx]:
                        self.impl.registers[idx] = rho
                return self

            if self.mode == 'sparse' and hll2.mode == 'dense':
                self.convert_to_dense()
                for i in range(m):
                    self.impl.registers[i] = max(
                        self.impl.registers[i], hll2.impl.registers[i]
                    )
                return self

            if self.mode == 'sparse' and hll2.mode == 'sparse':
                self_sparse = list(self.registers)
                other_sparse = list(hll2.registers)
                self_sparse.extend(other_sparse)

                deduped = {}
                for idx, rho in self_sparse:
                    if idx not in deduped or rho > deduped[idx]:
                        deduped[idx] = rho

                self.registers = sorted(deduped.items())

                if len(self.registers) > (m * (7.0 / 8)):
                    self.convert_to_dense()
                    return self.merge(hll2)

                return self
    def to_bytes(self) -> bytes:
        """
        Serialize this HLL into a stable, self-describing binary format.
        Layout: b"HLL1" | b | mode(0/1) | uint32_be(len) | payload(storing()).
        """
        magic = b"HLL1"
        if not (0 <= self.b < 32):
            raise ValueError(f"precision b out of range: {self.b}")
        mode_flag = 0 if self.mode == "dense" else 1
        payload = self.storing()  
        header = magic + bytes([self.b, mode_flag]) + struct.pack(">I", len(payload))
        return header + payload

    @classmethod
    def from_bytes(cls, blob: bytes) -> "HyperLogLog":
        """
        Reconstruct an HLL safely from the binary format produced by to_bytes().
        """
        if len(blob) < 10:  # 4+1+1+4
            raise ValueError("HLL blob too short")

        magic = blob[:4]
        if magic != b"HLL1":
            raise ValueError("Invalid HLL magic/version")

        b_val = blob[4]
        if not (0 <= b_val < 32):
            raise ValueError(f"Invalid precision b: {b_val}")

        mode_flag = blob[5]
        if mode_flag not in (0, 1):
            raise ValueError("Invalid mode flag")
        mode = "dense" if mode_flag == 0 else "sparse"

        (length,) = struct.unpack(">I", blob[6:10])
        if len(blob) != 10 + length:
            raise ValueError("Invalid HLL payload length")

        payload = blob[10:10+length]

   
        return cls(b=b_val, mode=mode, register=payload)


    def to_base64(self) -> str:
        """
        Return Base64 (no newlines) of to_bytes(). Compatible with decoders that ignore whitespace.
        """
        return base64.b64encode(self.to_bytes()).decode("ascii")

    @classmethod
    def from_base64(cls, s: str) -> "HyperLogLog":
        """
        Build from Base64 string (whitespace-tolerant).
        """
        data = base64.b64decode(s)  # ignores spaces/newlines like C decoder
        return cls.from_bytes(data)
        
        








