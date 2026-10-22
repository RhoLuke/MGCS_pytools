import numpy as np
from math import modf

from pandas import DataFrame

import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from shapely import (
    Polygon,
    Point,
    union,
    difference,
    intersection,
)

from scipy.spatial import ConvexHull

from .dr_correction import compute_differential_reddening
from .cvt_grid import cvtGrid
from .region_tools import get_regions_count, get_pts_in_regions

from mgcs_pytools.utils import plotting as utplot

#
#
# !!!!!!!! THIS CODE IS STILL UNDER DEVELOPMENT !!!!!!!!
#
#


def spatial_groupby(
    df_cluster: DataFrame, df_field: DataFrame, racol: str, deccol: str, minstars=200
):
    """Group stars in the cluster dataframe by their spatial location.
    The grouping is done by creating concentric annuli around the center of the
    cluster and dividing each annulus in regions with at least minstars stars.

    Args:
        df_cluster: DataFrame
            DataFrame containing the cluster stars.
        df_field: DataFrame
            DataFrame containing the field stars.
        racol: str
            Name of the column containing the RA of the stars.
        deccol: str
            Name of the column containing the DEC of the stars.
        minstars: int, optional
            Minimum number of stars in each region. Default is 200.

    Returns:
        groups: list of DataFrame
            List of DataFrames, each containing the stars in a region.
        subfovs: list of Polygon
            List of Polygons, each containing the area of a region.
        fov: Polygon
            Polygon containing the area of the instrument FoV.
    """

    # Calculate the field star density and the min area for at least min stars
    ra_center = (
        df_field[racol].min() + (df_field[racol].max() - df_field[racol].min()) / 2
    )
    dec_center = (
        df_field[deccol].min() + (df_field[deccol].max() - df_field[deccol].min()) / 2
    )

    field_center = SkyCoord(ra_center, dec_center, unit=u.deg)

    df_field.loc[:, "center_sep"] = field_center.separation(
        SkyCoord(
            df_field[racol],
            df_field[deccol],
            unit="deg",
        )
    ).arcsec

    area_conversion = 3600**2
    star_density = df_field.shape[0] / (np.pi * df_field["center_sep"].max() ** 2)
    min_stars = minstars
    min_area = min_stars / star_density

    print(
        f"Star density: {star_density:.2f} stars / arcsec^2",
        f"Min area for at least {min_stars} stars: {min_area:.2f} arcsec^2",
    )

    # groupby the cluster datafram in bin with at least minstars
    ra_center = (
        df_cluster[racol].min()
        + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    )
    dec_center = (
        df_cluster[deccol].min()
        + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    # make the FoV polygon from coordinates
    hull = ConvexHull(df_cluster.loc[:, [racol, deccol]].values)
    fov = Polygon(hull.points[hull.vertices])

    dr = 1 / 3600
    r = dr
    subfovs = [Point(ra_center, dec_center)]

    # iterating by increasing the annulus radius
    while True:
        # make annulus by subtraction wrt the last subfov
        subfov = Point(ra_center, dec_center).buffer(r)

        if subfov.contains(fov):
            break  # max fov exceeded
        elif (
            intersection(difference(subfov, subfovs[-1]), fov).area
            > min_area / area_conversion
        ):
            intersection(difference(subfov, subfovs[-1]), fov)
            subfovs.append(subfov)
        r += dr

    subfovs[-1] = union(intersection(fov, subfovs[-1]), difference(fov, subfovs[-1]))
    subfovs = [difference(subfovs[i], subfovs[i - 1]) for i in range(1, len(subfovs))]

    # TODO: This part need to be change using pandas groupby
    groups = []
    for subf in subfovs:
        group = df_cluster[
            df_cluster.apply(lambda x: Point(x[racol], x[deccol]).within(subf), axis=1)
        ].copy()
        groups.append(group)

    return groups, subfovs, fov


def get_membership(
    cell_counts: list[tuple[int, int]],
    cell_points: list[tuple[int, list[Point]]],
    common_regions: list[int],
    iter=1000,
    fov_ratio=1.0,
) -> np.ndarray:
    """Get the membership of each stars thorugh iterative random extractions.

    Args:
        cell_counts: list of tuple(int, int)
            List of tuples containing the region id and the number of field stars
            in that region.
        cell_points: list of tuple(int, list of Point)
            List of tuples containing the region id and the list of Points
            corresponding to the cluster stars in that region.
        common_regions: list of int
            List of region ids that are common between the field and cluster
            stars.
        iter: int, optional
            Number of iterations to perform. Default is 1000.
        fov_ratio: float, optional
            Ratio between the instrument FoV and the sub-FoV area.
            Default is 1.0.

    Returns:
        membership: np.ndarray
            Array of shape (N, 2) containing the star index and its membership
            probability.
    """

    cell_counts_points = [
        (
            [count for rrid, count in cell_counts if rrid == rid][0],
            [pts for rrid, pts in cell_points if rrid == rid][0],
        )
        for rid in common_regions
    ]

    # initialize extractions vector
    extractions = np.full(
        iter * np.sum([count for count, _ in cell_counts_points]),
        fill_value=-1,
        dtype=int,
    )

    c = 0
    for _ in range(iter):
        for count, pts in cell_counts_points:
            # rescaling the counts
            new_count = count * fov_ratio
            if new_count < 1.0:
                new_count = np.random.choice(
                    np.array([0, 1]), p=[1 - new_count, new_count]
                )
            else:
                p, bias = modf(new_count)
                new_count = int(bias) + np.random.choice(np.array([0, 1]), p=[1 - p, p])

            if len(pts) > new_count:
                extractions[c : c + new_count] = np.random.choice(
                    pts, new_count, replace=False
                )
                c += new_count
            else:
                extractions[c : c + (len(pts))] = list(pts)
                c += len(pts)

    extractions = extractions[extractions != -1]
    extractions = sorted(extractions)
    unique_id = np.unique(extractions)
    bincount = np.bincount(extractions)
    bincount = bincount[bincount > 0]

    membership = np.vstack((unique_id, bincount)).T
    res = [1 - count / iter for _, count in membership]

    membership = np.array(membership).astype(float)
    membership[:, 1] = res

    return membership


def do_statistical_membership(
    df_cluster_input: DataFrame,
    df_field_input: DataFrame,
    target_mag_col: list[str],
    target_mag_col_error: list[str],
    field_mag_col: list[str],
    dr_params: dict[str, str],
    parallel_fov: float,
    member_threshold: float = 0.8,
    minstars: int = 200,
    racol: str = "RA",
    deccol: str = "DEC",
    # process_iter: int = 3,
    min_star_per_cell: int = 3,
    membership_iter: int = 1000,
    roi: list[float] | None = None,
    plot_dred: bool = False,
    plot_voronoi: bool = False,
    which_voronoi: str = "dilation",
    do_dilation: bool = True,
) -> DataFrame:
    """Compute the statistical membership of stars in a cluster field.

    Args:
        df_cluster_input: DataFrame
            DataFrame containing the cluster stars.
        df_field_input: DataFrame
            DataFrame containing the field stars.
        target_mag_col: list of str
            List containing the names of the columns with the magnitudes
            to use for the target CMD (color, mag).
        target_mag_col_error: list of str
            List containing the names of the columns with the magnitude errors
            to use for the target CMD (color, mag).
        field_mag_col: list of str
            List containing the names of the columns with the magnitudes
            to use for the field CMD (color, mag).
        dr_params: dict of str
            Dictionary containing inputs parameters for the differential reddening
            correction. It must contain the following keys:
            - 'rband1': the exction value in band1,
            - 'rband2': the exction value in band2,
            - 'TO_mag': the magnitude of the Turn-off,
            - 'TO_color': the color of the Turn-off,
            - 'nref': the number of reference stars,
            - 'xcol': the name of the column containing the x coordinates,
            - 'ycol': the name of the column containing the y coordinates,
            - 'ord_step': the bin size in the 'ordinate' to calculate the ridge line
        parallel_fov: float
            Size of the parallel FoV in deg square.
        member_threshold: float, optional
            Membership threshold to consider a star as a member of the cluster.
            Default is 0.8.
        minstars: int, optional
            Minimum number of stars in each region. Default is 200.
        racol: str, optional
            Name of the column containing the RA of the stars. Default is "RA".
        deccol: str, optional
            Name of the column containing the DEC of the stars. Default is "DEC".
        min_star_per_cell: int, optional
            Minimum number of stars per Voronoi cell. Default is 3.
        membership_iter: int, optional
            Number of iterations to perform for the membership calculation.
            Default is 1000.
        fov_ratio: float, optional
            Ratio between the instrument FoV and the sub-FoV area.
            Default is 1.0.
        roi: list of float or None, optional
            List containing the vertices of a polygon to use as region of interest
            for the differential reddening correction.
            Example: [x1, y1, x2, y2, ..., xn, yn]
            If None, no region of interest will be used. Default is None.
        plot_dred: bool, optional
            If True, plot the differential reddening correction results.
            Default is False.
        plot_voronoi: bool, optional
            If True, plot the Voronoi grid used for the membership calculation.
            Default is False.
        which_voronoi: str, optional
            Type of Voronoi grid to use. Options are "standard" and "dilation".
            Default is "dilation".
        do_dilation: bool, optional
            If True, dilate the Voronoi regions to have at least min_star_per_cell
            stars in each cell. Default is True.

    Returns:
        df_cluster: DataFrame
            DataFrame containing the cluster stars with the membership and
            differential reddening corrected magnitudes added.
    """
    # make a copy to preserve the original dataframe
    df_cluster = df_cluster_input.copy()
    df_field = df_field_input.copy()

    groups, subfovs, instrument_fov = spatial_groupby(
        df_cluster,
        df_field,
        racol=racol,
        deccol=deccol,
        minstars=minstars,
    )

    ra_center = (
        df_cluster[racol].min()
        + (df_cluster[racol].max() - df_cluster[racol].min()) / 2
    )
    dec_center = (
        df_cluster[deccol].min()
        + (df_cluster[deccol].max() - df_cluster[deccol].min()) / 2
    )

    fov_ratio = [subfov.area / parallel_fov for subfov in subfovs]

    field_pts = [
        Point(row[field_mag_col[0]] - row[field_mag_col[1]], row[field_mag_col[0]])
        for _, row in df_field.iterrows()
    ]

    # initialize the cluster columns for membership and differential reddening
    dr_band1_corr = target_mag_col[0] + "_drcorr"
    dr_band2_corr = target_mag_col[1] + "_drcorr"
    df_cluster[dr_band1_corr] = df_cluster[target_mag_col[0]]
    df_cluster[dr_band2_corr] = df_cluster[target_mag_col[1]]

    dr_params["band1"] = dr_band1_corr
    dr_params["band2"] = dr_band2_corr

    dr_params["xcol"] = racol
    dr_params["ycol"] = deccol

    df_cluster["delta_ebv"] = 0.0
    df_cluster["membership"] = 1.0

    it = 0
    it_stop = None
    # for it in range(process_iter):
    while True:
        print(f"ITERATION {it + 1}")

        # reddening differential correction
        print("Compute differential reddening..")
        # print(df_cluster["membership"].describe())
        if it == 0:
            print("First iteration, correction threshold skipped")
            calc_correction_threshold = False
        else:
            print("Do calculation of the correction threshold")
            calc_correction_threshold = True

        b1_corr, b2_corr, EBV, poli = compute_differential_reddening(
            df_cluster,
            params=dr_params,
            member_threshold=member_threshold,
            calc_correction_threshold=calc_correction_threshold,
            roi=roi,
            plot=plot_dred,
        )

        roi = poli
        df_cluster[dr_band1_corr] = b1_corr
        df_cluster[dr_band2_corr] = b2_corr
        df_cluster["delta_ebv"] += EBV

        if plot_dred:
            # plot new cmd
            _, ax = plt.subplots(layout="constrained")

            utplot.plot_cmd(
                df_cluster[dr_band1_corr],
                df_cluster[dr_band2_corr],
                ax,
            )
            # ax.set(xlim=(0.5, 3.5), ylim=(26.5, 14))

            # plot reddening map
            _ = utplot.plot_reddening_map(
                df_cluster[racol],
                df_cluster[deccol],
                ra_center,
                dec_center,
                EBV,
            )
            # plt.show(block=False)
            plt.draw()
            plt.pause(1.0)

        print("...differential reddening completed")
        print(
            f"EBV stat (mean, min, max): {np.median(EBV):.6f}, {EBV.min():.6f}, {EBV.max():.6f}"
        )
        # _, ax = plt.subplots()
        # ax.hist(EBV, bins=30, color="gray", edgecolor="black", density=True)
        # plt.show()
        print(
            f"Number of stars with null ebv: {np.sum(EBV == 0.0) / EBV.shape[0] * 100:.2f}%"
            f" ({np.sum(EBV == 0.0)} over {EBV.shape[0]})"
        )
        photometric_error = np.median(
            (
                df_cluster[target_mag_col_error[0]] ** 2
                + df_cluster[target_mag_col_error[1]] ** 2
            )
            ** 0.5
        )
        print(f"Mean photometric error: {photometric_error:.6f}")

        print("Start decontamination process..")
        for i, (group, fovr) in enumerate(zip(groups, fov_ratio)):
            # updating the membership of the groups
            # !!! this can be avoided if pandas groupby is used !!!
            group.loc[:, "membership"] = df_cluster.loc[group.index, "membership"]
            group.loc[:, dr_band1_corr] = df_cluster.loc[group.index, dr_band1_corr]
            group.loc[:, dr_band2_corr] = df_cluster.loc[group.index, dr_band2_corr]

            print(f"\t#### GROUP N.{i + 1} ####")

            # Voronoi grid creation
            xg = group[dr_band1_corr] - group[dr_band2_corr]
            yg = group[dr_band1_corr]
            cluster_pts = [Point(x, y) for x, y in zip(xg, yg)]

            # dilate regions to have at least n stars in each cell
            if do_dilation:
                if min_star_per_cell > group.shape[0]:
                    print(
                        f"\tNot enough stars in this region "
                        f"({group.shape[0]}, requested {min_star_per_cell})"
                    )
                    dilate = False
                else:
                    dilate = True
            else:
                dilate = False

            # cvt_grid = cvtGrid(points, iter=1, dilate=dilate)

            cvt_grid = cvtGrid(
                np.array([xg.values, yg.values]).T,
                iter=1,
                dilate=dilate,
                which=which_voronoi,
                target_median=min_star_per_cell,
            )

            regions = cvt_grid.grid if not dilate else cvt_grid.dilated_grid
            cluster_ids = group.index.values

            # count field stars in cells
            cell_field_counts = get_regions_count(regions, field_pts)
            points_in_reigons = get_pts_in_regions(regions, cluster_pts)

            if plot_voronoi:
                _ = utplot.plot_cmd_and_vorgrid(
                    group,
                    [dr_band1_corr, dr_band2_corr],
                    regions,
                    points_in_reigons,
                )
                # plt.show(block=False)
                plt.draw()
                plt.pause(1.0)

            common_regions = set([el[0] for el in cell_field_counts]).intersection(
                set([el[0] for el in points_in_reigons])
            )

            # add membership
            membership = get_membership(
                cell_field_counts,
                points_in_reigons,
                common_regions,
                membership_iter,
                fovr,
            )

            print(
                f"\tMembership (mean, min, max): "
                f"({membership[:, 1].mean():.4f},"
                f" {membership[:, 1].min():.4f},"
                f" {membership[:, 1].max():.4f})\n\n"
            )

            df_cluster.loc[cluster_ids[membership[:, 0].astype(int)], "membership"] = (
                membership[:, 1]
            )

        if it == it_stop:
            break

        if abs(np.median(EBV)) < abs(photometric_error) and it > 0:
            print("Differential reddening converged, stopping iterations.")
            print("Next iteration will be the last one.")
            it_stop = it + 1

        it += 1

    return df_cluster
