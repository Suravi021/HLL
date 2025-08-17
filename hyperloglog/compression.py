import numpy as np
from numpy.typing import NDArray

def pack_registers(registers: NDArray[np.int32], binbits: int) -> bytes:

    """
    Packs a list of integer registers into a bytes object using the specified number of bits per register.
    
    Args:
        registers: List[int] - register values to pack (must be non-negative)
        binbits: int - number of bits per register (must be positive, max 64 for safety)
    
    Returns:
        bytes: packed register data
        
    Raises:
        ValueError: If inputs are invalid
        OverflowError: If the operation would cause memory issues
    """

    # Input validation
    if not (isinstance(registers, np.ndarray) and np.issubdtype(registers.dtype, np.integer)):
        raise ValueError("registers must be a numpy array of integers")

    if not isinstance(binbits, int) or binbits <= 0:
        raise ValueError("binbits must be a positive integer")
    
    if binbits > 64:
        raise ValueError("binbits must be <= 64 to prevent memory issues")
    
    if not registers:
        return b''
    
    # Check register values
    max_val = (1 << binbits) - 1

    for i, val in enumerate(registers):

        if not isinstance(val, int):
            raise ValueError(f"Register {i} must be an integer")
        
        if val < 0:
            raise ValueError(f"Register {i} must be non-negative")
        
        if val > max_val:
            raise ValueError(f"Register {i} value {val} exceeds {binbits}-bit limit ({max_val})")
    
    # Check for potential overflow
    total_bits = len(registers) * binbits
    if total_bits > 2**20:  # Arbitrary safety limit (1MB of bits)
        raise OverflowError(f"Total bits ({total_bits}) too large, risk of memory overflow")
    
    # Pack registers using bitwise operations
    m = len(registers)
    bitstream = 0
    for i, val in enumerate(registers):
        bitstream |= (val & ((1 << binbits) - 1)) << (i * binbits)
    
    needed_bytes = (m * binbits + 7) // 8
    return bitstream.to_bytes(needed_bytes, byteorder='little')


def unpack_registers(data: bytes, m: int, binbits: int) -> NDArray[np.int32]:
    """
    Unpacks a bytes object into a list of integer registers using the specified number of bits per register.
    
    Args:
        data: bytes - packed register data
        m: int - number of registers (must be non-negative)
        binbits: int - number of bits per register (must be positive)
    
    Returns:
        List[int]: unpacked register values
        
    Raises:
        ValueError: If inputs are invalid or data is insufficient
    """

    # Input validation
    if not isinstance(data, bytes):
        raise ValueError("data must be bytes")
    if not isinstance(m, int) or m < 0:
        raise ValueError("m must be a non-negative integer")
    if not isinstance(binbits, int) or binbits <= 0:
        raise ValueError("binbits must be a positive integer")
    if binbits > 64:
        raise ValueError("binbits must be <= 64 to prevent memory issues")
    if m == 0:
        return np.array([], dtype=np.int32)
    
    # Check if we have enough data
    required_bits = m * binbits
    required_bytes = (required_bits + 7) // 8
    if len(data) < required_bytes:
        raise ValueError(f"Insufficient data: need {required_bytes} bytes, got {len(data)}")
    
    # Check for potential overflow
    if required_bits > 2**20:  # Same safety limit as pack
        raise OverflowError(f"Total bits ({required_bits}) too large, risk of memory overflow")
    
    # Unpack registers
    bitstream = int.from_bytes(data, byteorder='little')
    regs = np.empty(m, dtype=np.int32)
    mask = (1 << binbits) - 1
    
    for i in range(m):
        shift = i * binbits
        regs[i] = (bitstream >> shift) & mask
    
    return regs

def compress_sparse_registers(sparse_registers: NDArray[np.int32], b: int, rbits : int = 6) -> bytes:
    """
    Compresses sparse HLL registers (list of (idx, rho)) into a bytes object.
    Args:
        sparse_registers: NDArray[np.int32], shape (n, 2)
        b: int - number of bits for the index
        rbits: int - number of bits for rho (default 6)
    Returns:
        bytes: compressed sparse register representation
    """
    bitstream = 0
    total_bits = 0
    entrybits = b + rbits
    
    for idx, rho in sparse_registers:
        entry = (idx << rbits) | (rho & ((1 << rbits) - 1))
        bitstream |= entry << total_bits
        total_bits += entrybits

    num_bytes = (total_bits + 7) // 8
    return bitstream.to_bytes(num_bytes, byteorder='little')
    
def decompress_sparse_registers(data: bytes, b: int, rbits: int=6) -> NDArray[np.int32] :
    """
    Decompresses sparse HLL registers from a bytes object into a list of (idx, rho).
    Args:
        data: bytes - compressed sparse register data
        b: int - number of bits for the index
        rbits: int - number of bits for rho (default 6)
    Returns:
        NDArray[np.int32], shape (n, 2)
    """
    bitstream = int.from_bytes(data, byteorder='little')
    entrybits = b + rbits
    total_bits = len(data) * 8
    num_entries = total_bits // entrybits

    sparse_registers = np.empty((num_entries, 2), dtype=np.int32)

    mask = (1 << entrybits) - 1
    rhomask = (1 << rbits) - 1

    for i in range(num_entries):
        shift = i * entrybits
        entry = (bitstream >> shift) & mask
        idx = entry >> rbits
        rho = entry & rhomask
        sparse_registers[i, 0] = idx
        sparse_registers[i, 1] = rho
    
    return sparse_registers

