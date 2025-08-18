
import numpy as np
from typing import Union

def murmurhash64a(key: Union[str, bytes], seed: int = 0) -> int:
    """
        MurmurHash64A implementation ported from C (PostgreSQL).
        
        Args:
            key: The data to hash. Can be either a string (UTF-8 encoded) or bytes.
            seed: Initial seed value for the hash function. Defaults to 0.
                  Different seeds produce different hash values for the same input.
        
        Returns:
            A 64-bit integer hash value.
            
        Raises:
            TypeError: If key is neither str nor bytes.

    """
    if isinstance(key, str):
        key_bytes = key.encode('utf8')
    elif isinstance(key, bytes):
        key_bytes = key
    else:
        raise TypeError("key must be str or bytes")

    MULTIPLIER = np.uint64(0xc6a4a7935bd1e995)
    RIGHT_SHIFT = 47
    MASK_64BIT = np.uint64(0xFFFFFFFFFFFFFFFF)

    data_length = len(key_bytes)
    hash_value = np.uint64(seed) ^ (np.uint64(data_length) * MULTIPLIER)
    hash_value &= MASK_64BIT

    # Process full 8-byte chunks
    full_chunks_end = (data_length // 8) * 8
    if full_chunks_end:
        chunks = np.frombuffer(key_bytes[:full_chunks_end], dtype='<u8')
        k = (chunks * MULTIPLIER) & MASK_64BIT
        k ^= (k >> np.uint64(RIGHT_SHIFT))
        k = (k * MULTIPLIER) & MASK_64BIT

        for val in k:
            val = np.uint64(val)
            hash_value ^= val
            hash_value = (hash_value * MULTIPLIER) & MASK_64BIT

    # Handle remaining bytes
    remaining = data_length & 7
    if remaining:
        last_chunk = key_bytes[-remaining:] + b'\x00' * (8 - remaining)
        val = np.uint64(np.frombuffer(last_chunk, dtype='<u8')[0])
        hash_value ^= (val * MULTIPLIER) & MASK_64BIT
        hash_value = (hash_value * MULTIPLIER) & MASK_64BIT

    # Final avalanche
    hash_value ^= (hash_value >> np.uint64(RIGHT_SHIFT))
    hash_value = (hash_value * MULTIPLIER) & MASK_64BIT
    hash_value ^= (hash_value >> np.uint64(RIGHT_SHIFT))

    return int(hash_value)
