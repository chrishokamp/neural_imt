"""
Read a file and create a vocab dict from it -- ensure that BOS, EOS, and UNK tokens are always included,
even if they aren't in the input file

"""

# WORKING: use the arrow syntax like Sennrich's scripts

# these indexes are used for consistency with _ensure_special_tokens from the NMT repo
BOS_IDX = 0
UNK_IDX = 1
# EOS_IDX (index of the last word in the vocab)