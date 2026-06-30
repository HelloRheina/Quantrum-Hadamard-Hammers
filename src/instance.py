from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WirelessInstance:
    users: int
    channels: int
    demand: np.ndarray
    gain: np.ndarray
    energy: np.ndarray
    interference_edges: list
    channel_capacity: np.ndarray
    seed: int


def generate_instance(users=5, channels=3, seed=7, edge_prob=0.4):
    """Create a reproducible synthetic wireless channel-allocation instance."""
    rng = np.random.default_rng(seed)
    demand = rng.uniform(0.7, 1.5, users)
    gain = rng.uniform(0.4, 1.4, (users, channels))
    energy = rng.uniform(0.2, 1.2, (users, channels))
    edges = []
    for u in range(users):
        for v in range(u + 1, users):
            if rng.random() < edge_prob:
                edges.append((u, v, float(rng.uniform(0.2, 1.0))))
    if not edges and users > 1:
        edges.append((0, 1, float(rng.uniform(0.2, 1.0))))
    capacity = rng.uniform(1.5, 3.0, channels)
    return WirelessInstance(users, channels, demand, gain, energy, edges, capacity, seed)

