import numpy as np
from src.encoding import encode_assignment, decode_bitstring
from src.instance import generate_instance


def test_one_hot_encode_decode():
    assignment = np.array([0, 2, 1])
    bits = encode_assignment(assignment, 3)
    decoded, valid = decode_bitstring(bits, 3, 3)
    assert valid
    assert decoded.tolist() == assignment.tolist()


def test_repair_invalid_bitstring():
    inst = generate_instance(2, 3, seed=1)
    bits = np.array([0, 0, 0, 1, 1, 0])
    decoded, valid = decode_bitstring(bits, 2, 3, inst, repair=True)
    assert not valid
    assert decoded.shape == (2,)
    assert all(0 <= c < 3 for c in decoded)

