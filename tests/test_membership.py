from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
from mgcs_pytools.statistical_membership import (
    spatial_groupby,
    compute_statistical_membership,
)
from mgcs_pytools.statistical_membership.membership import get_membership


def test_spatial_groupby():
    np.random.seed(42)
    # mock cluster tight distribution
    df_cluster = pd.DataFrame(
        {"RA": np.random.normal(0, 0.05, 1000), "DEC": np.random.normal(0, 0.05, 1000)}
    )
    # mock field spread
    df_field = pd.DataFrame(
        {
            "RA": np.random.uniform(-0.5, 0.5, 5000),
            "DEC": np.random.uniform(-0.5, 0.5, 5000),
        }
    )

    groups, subfovs, fov = spatial_groupby(
        df_cluster, df_field, racol="RA", deccol="DEC", minstars=100
    )

    assert len(groups) > 0
    assert len(groups) == len(subfovs)
    total_stars = sum([len(g) for g in groups])
    assert total_stars == 1000


def test_get_membership_analytical_and_montecarlo():
    cell_counts = [(1, 10), (2, 5)]
    cell_points = [(1, list(range(10))), (2, list(range(10, 20)))]
    common_regions = [1, 2]

    mem_ana = get_membership(
        cell_counts, cell_points, common_regions, fov_ratio=0.5, method="analytical"
    )
    assert mem_ana.shape == (20, 2)
    # 10 field stars * 0.5 / 10 target stars = 0.5 field equivalent
    # prob = 1.0 - 0.5 = 0.5
    assert np.isclose(mem_ana[0, 1], 0.5)

    mem_mc = get_membership(
        cell_counts,
        cell_points,
        common_regions,
        iter=100,
        fov_ratio=0.5,
        method="montecarlo",
    )
    assert mem_mc.shape == (20, 2)
    assert np.all((mem_mc[:, 1] >= 0.0) & (mem_mc[:, 1] <= 1.0))
    # MC should be roughly around the analytical value
    assert np.abs(np.mean(mem_mc[:10, 1]) - 0.5) < 0.2




def test_compute_statistical_membership():
    np.random.seed(42)
    df_cluster = pd.DataFrame(
        {
            "RA": np.random.normal(0, 0.05, 200),
            "DEC": np.random.normal(0, 0.05, 200),
            "mag1": np.random.uniform(15, 20, 200),
            "mag2": np.random.uniform(14, 19, 200),
            "mag1_err": np.random.uniform(0.01, 0.05, 200),
            "mag2_err": np.random.uniform(0.01, 0.05, 200),
        }
    )
    df_field = pd.DataFrame(
        {
            "RA": np.random.uniform(-0.5, 0.5, 500),
            "DEC": np.random.uniform(-0.5, 0.5, 500),
            "mag1": np.random.uniform(15, 20, 500),
            "mag2": np.random.uniform(14, 19, 500),
        }
    )

    dr_params = {
        "rband1": 3.1,
        "rband2": 2.5,
        "TO_mag": 17.0,
        "TO_color": 1.0,
        "nref": 20,
        "ord_step": 0.5,
    }

    # We patch the dr correction so it converges immediately
    with patch(
        "mgcs_pytools.statistical_membership.membership.compute_differential_reddening"
    ) as mock_dr:
        mock_dr.return_value = (
            df_cluster["mag1"],
            df_cluster["mag2"],
            np.zeros(200),
            None,
        )

        try:
            res = compute_statistical_membership(
                df_cluster,
                df_field,
                target_mag_col=["mag1", "mag2"],
                target_mag_col_error=["mag1_err", "mag2_err"],
                field_mag_col=["mag1", "mag2"],
                dr_params=dr_params,
                parallel_fov=1.0,
                minstars=50,
                min_star_per_cell=2,
                membership_iter=10,
            )
            assert "membership" in res.columns
            assert "delta_ebv" in res.columns
        except Exception as e:
            pytest.fail(f"compute_statistical_membership raised an exception: {e}")
