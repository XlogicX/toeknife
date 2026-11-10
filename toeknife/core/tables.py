"""Constants from RFC 1951 (DEFLATE Compressed Data Format Specification v1.3)."""

# Section 3.2.5 -- length codes 257..285
LENGTH_BASE = (
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
)
LENGTH_EXTRA = (
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
)

# Section 3.2.5 -- distance codes 0..29.  Codes 30 and 31 never occur in valid
# data, but they *are* representable in fixed mode, which makes them a useful
# misalignment signal.
DISTANCE_BASE = (
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
)
DISTANCE_EXTRA = (
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
)

# Section 3.2.7 -- the shuffled order code-length code lengths arrive in
CODE_LENGTH_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)

# Section 3.2.6 -- fixed Huffman code lengths.
#
# The distance table is 32 entries, not 29.  The original ToeKnife built only
# codes 0..28, which made distance code 29 (24577..32768 bytes back) silently
# undecodable.  Codes 30/31 are included so that hitting one is reported as an
# invalid symbol rather than a decode failure.
FIXED_LITLEN_LENGTHS = tuple([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
FIXED_DIST_LENGTHS = tuple([5] * 32)

MAX_CODE_BITS = 15          # no DEFLATE Huffman code exceeds 15 bits
MAX_WINDOW = 32768          # LZ77 sliding window
END_OF_BLOCK = 256

BTYPE_STORED = 0
BTYPE_FIXED = 1
BTYPE_DYNAMIC = 2
BTYPE_RESERVED = 3

BTYPE_NAMES = {
    BTYPE_STORED: "stored",
    BTYPE_FIXED: "fixed",
    BTYPE_DYNAMIC: "dynamic",
    BTYPE_RESERVED: "reserved(invalid)",
}

# Longest possible single symbol: 15-bit length code + 5 extra + 15-bit distance
# code + 13 extra.  Used to bound how far a misaligned decode can be dragged.
MAX_SYMBOL_BITS = 48
