import numpy as np
import pandas as pd
from metrics import hv_at_common_budget, auc_hv_on_grid, interpolate_curve, normalized_hypervolume_mc


def test_common_grid_interpolation_progressive():
    curve = pd.DataFrame({"cumulative_evaluations": [10, 20], "hv": [1.0, 3.0]})
    grid = np.array([0, 10, 15, 20])
    vals = interpolate_curve(curve, grid)
    assert vals[0] == 0.0
    assert vals[-1] == 3.0
    assert auc_hv_on_grid(curve, grid) > 0


def test_hv_at_common_budget_keys():
    curve = pd.DataFrame({"cumulative_evaluations": [10, 20], "hv": [1.0, 3.0]})
    out = hv_at_common_budget(curve, 20)
    assert set(out) == {"hv_at_25pct", "hv_at_50pct", "hv_at_75pct", "hv_at_100pct"}


def test_normalized_hv_is_scale_invariant():
    points = np.array([[1.0, 2.0, 3.0], [2.0, 1.5, 2.0]])
    ref = np.array([4.0, 5.0, 6.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([4.0, 5.0, 6.0])
    hv = normalized_hypervolume_mc(points, ref, lo, hi, samples=5000, seed=7)
    scale = np.array([10.0, 100.0, 1000.0])
    scaled_hv = normalized_hypervolume_mc(points * scale, ref * scale, lo * scale, hi * scale, samples=5000, seed=7)
    assert np.isclose(hv, scaled_hv)
