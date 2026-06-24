import numpy as np
import pandas as pd

from scripts.analyze_deng_frozen_mature_dqn import (
    exact_two_sided_sign_test,
    paired_sign_flip_p_value,
    permutation_p_value,
)
from scripts.reproduce_tables import exact_permutation_p_value, holm_adjust


def test_exact_two_sample_permutation_matches_enumeration():
    left = np.array([2.0, 2.0])
    right = np.array([0.0, 0.0])

    assert exact_permutation_p_value(left, right) == 2.0 / 6.0
    assert permutation_p_value(left, right) == 2.0 / 6.0


def test_exact_paired_sign_flip_uses_checkpoint_differences():
    assert paired_sign_flip_p_value(np.array([-1.0, -1.0])) == 0.5


def test_exact_two_sided_sign_test_uses_directions_only():
    assert exact_two_sided_sign_test(np.array([1.0] * 9 + [-100.0])) == 22.0 / 1024.0
    assert exact_two_sided_sign_test(np.array([-1.0] * 10)) == 2.0 / 1024.0


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"])
    adjusted = holm_adjust(raw)

    assert adjusted["a"] == 0.03
    assert adjusted["c"] == 0.06
    assert adjusted["b"] == 0.06
