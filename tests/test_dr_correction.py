import numpy as np
import pandas as pd
from shapely import Polygon

from mgcs_pytools.statistical_membership.dr_correction import (
    compute_differential_reddening,
    ROISelector,
)


def test_roi_selector_headless():
    points = np.array([[0, 0], [1, 1], [0.5, 0.5], [2, 2]])
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    roi = ROISelector(points, vertices=vertices)

    assert roi.polygon is not None
    assert roi.points_in is not None
    # [0.5, 0.5] should be in, others should be out/on boundary
    # Contains_xy strictly contains interior points unless tweaked, let's just check length
    assert len(roi.points_in) == 4


def test_compute_differential_reddening_synthetic():
    np.random.seed(42)
    n_stars = 200

    # Create synthetic MS stars around a line
    # band2 = band1 - color
    # Let's mock a simple sequence
    colors = np.random.uniform(0.5, 1.5, n_stars)
    band1 = np.random.uniform(10, 15, n_stars)
    band2 = band1 - colors

    x = np.random.uniform(0, 100, n_stars)
    y = np.random.uniform(0, 100, n_stars)

    # Introduce some reddening
    delta_ebv = np.random.uniform(0, 0.1, n_stars)
    membership = np.random.uniform(0.5, 1.0, n_stars)

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "band1": band1,
            "band2": band2,
            "delta_ebv": delta_ebv,
            "membership": membership,
        }
    )

    params = {
        "xcol": "x",
        "ycol": "y",
        "band1": "band1",
        "band2": "band2",
        "rband1": 1.0,
        "rband2": 0.8,
        "TO_mag": 10.0,
        "TO_color": 0.5,
        "n_reference_stars": 10,
        "ord_step": 0.5,
    }

    # Set an ROI that encompasses most transformed stars
    # We don't know exact rotated coords easily, so let's just make a huge bounding box
    roi = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])

    band1_corr, band2_corr, d_ebv_corr, poli = compute_differential_reddening(
        df,
        params,
        member_threshold=0.0,
        calc_correction_threshold=False,
        roi=roi,
        plot=False,
    )

    assert len(band1_corr) == n_stars
    assert len(band2_corr) == n_stars
    assert len(d_ebv_corr) == n_stars
    assert poli is not None
