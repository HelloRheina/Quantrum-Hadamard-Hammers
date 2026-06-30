import numpy as np


def var_index(user, channel, channels):
    return user * channels + channel


def encode_assignment(assignment, channels):
    bits = np.zeros(len(assignment) * channels, dtype=np.int8)
    for u, c in enumerate(assignment):
        bits[var_index(u, int(c), channels)] = 1
    return bits


def decode_bitstring(bits, users, channels, instance=None, repair=True):
    bits = np.asarray(bits, dtype=np.int8)
    assignment = []
    valid = True
    for u in range(users):
        block = bits[u * channels:(u + 1) * channels]
        ones = np.flatnonzero(block == 1)
        if len(ones) == 1:
            assignment.append(int(ones[0]))
        else:
            valid = False
            if not repair:
                assignment.append(-1)
            elif block.sum() > 0:
                assignment.append(int(np.argmax(block)))
            elif instance is not None:
                score = instance.demand[u] * instance.gain[u] - 0.25 * instance.energy[u]
                assignment.append(int(np.argmax(score)))
            else:
                assignment.append(0)
    return np.asarray(assignment, dtype=int), valid


def assignment_to_label(assignment):
    return "-".join(str(int(c)) for c in assignment)


def enumerate_valid_assignments(users, channels):
    grids = np.indices((channels,) * users).reshape(users, -1).T
    return grids.astype(int)


def bitstrings_to_matrix(n_qubits):
    values = np.arange(2 ** n_qubits, dtype=np.uint32)
    shifts = np.arange(n_qubits, dtype=np.uint32)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.int8)

