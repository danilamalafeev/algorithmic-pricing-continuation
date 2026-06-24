import pandas as pd
import pytest

from scripts.compute_binned_trajectory_welfare import consumer_surplus, weighted_window


def test_consumer_surplus_falls_when_both_prices_rise():
    assert consumer_surplus(1.5, 1.5) > consumer_surplus(1.8, 1.8)


def test_weighted_window_uses_bin_overlap():
    frame = pd.DataFrame(
        {
            "bin_start": [0, 40, 80],
            "bin_end": [40, 80, 100],
            "bin_steps": [40, 40, 20],
            "total_steps": [100, 100, 100],
            "avg_price_oracle": [1.0, 2.0, 3.0],
            "avg_price_victim": [1.0, 2.0, 3.0],
            "avg_profit_oracle": [0.1, 0.2, 0.3],
            "avg_profit_victim": [0.1, 0.2, 0.3],
            "joint_profit": [0.2, 0.4, 0.6],
            "consumer_surplus_approx": [0.9, 0.6, 0.3],
            "total_welfare_approx": [1.1, 1.0, 0.9],
        }
    )

    selected, metrics = weighted_window(frame, 60)

    assert selected["overlap_steps"].tolist() == [40, 20]
    assert metrics["covered_steps"] == 60
    assert metrics["avg_price_oracle"] == pytest.approx((2.0 * 40 + 3.0 * 20) / 60)
    assert metrics["total_welfare_approx"] == pytest.approx((1.0 * 40 + 0.9 * 20) / 60)
