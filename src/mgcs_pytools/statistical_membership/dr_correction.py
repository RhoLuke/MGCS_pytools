import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely import (
    Polygon,
    plotting,
)

from scipy.interpolate import CubicSpline


import shapely
from typing import Optional, List, Tuple


class ROISelector:
    def __init__(
        self, points: np.ndarray, vertices: Optional[List[Tuple[float, float]]] = None
    ):
        self.points = points

        self.ax = None
        self.fig = None
        self.cid_MOUSE = None
        self.cid_KEYBOARD = None
        self.coords = vertices if vertices is not None else []
        self.polygon = None
        self.points_in = None
        self.patch = None

        if len(self.coords) >= 3:
            self.polygon = Polygon(self.coords)
            self.points_in = shapely.contains_xy(
                self.polygon, self.points[:, 0], self.points[:, 1]
            )

    def start_selection(self):
        # generate fig
        self.generate_fig()

        # make plot
        self.ax.scatter(self.points[:, 0], self.points[:, 1], marker=".", s=0.5, lw=0.1)
        self.ax.set(
            xlabel="abscissa",
            ylabel="ordinate",
            title="CMD in the abscissa-ordinate plane",
        )
        self.ax.text(
            0.05,
            0.95,
            "Left click: add vertex\nSpacebar: close polygon\nC: cancel polygon\nQ: quit",
            transform=self.ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        self.fig.canvas.draw()
        plt.show()

    def __call__(self, event):

        # left click add vertices
        if event.name == "button_press_event" and event.button == 1:
            self.coords.append((event.xdata, event.ydata))
            self.ax.scatter(
                self.coords[-1][0], self.coords[-1][1], marker="o", c="blue", zorder=3
            )

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # spacebar create polygon
        elif event.name == "key_press_event" and event.key == " ":
            self.coords.append(self.coords[0])  # close polygon
            self.polygon = Polygon(self.coords)  # close polygon
            self.points_in = shapely.contains_xy(
                self.polygon, self.points[:, 0], self.points[:, 1]
            )
            self.ax.scatter(
                self.points[self.points_in, 0],
                self.points[self.points_in, 1],
                marker=".",
                s=0.5,
                lw=0.5,
                c="red",
                zorder=3,
            )
            self.patch = plotting.patch_from_polygon(
                self.polygon, color="gray", alpha=0.5
            )

            self.ax.add_patch(self.patch)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # c cancel the polygon
        elif event.name == "key_press_event" and event.key == "c":
            self.coords = []
            self.polygon = None
            self.patch = None
            self.ax.clear()
            self.ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                marker=".",
                s=0.5,
                lw=0.5,
                zorder=3,
            )

            self.ax.set(
                xlabel="abscissa",
                ylabel="ordinate",
                title="CMD in the abscissa-ordinate plane",
            )
            self.ax.text(
                0.05,
                0.95,
                "Left click: add vertex\nSpacebar: close polygon\nC: cancel polygon\nQ: quit",
                transform=self.ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        # q close the intercative session
        elif event.name == "key_press_event" and event.key == "q":
            for c in [self.cid_KEYBOARD, self.cid_MOUSE]:
                self.fig.canvas.mpl_disconnect(c)
            plt.close(self.fig)
            return

    def generate_fig(self):
        self.fig, self.ax = plt.subplots(layout="constrained")
        self.cid_MOUSE = self.ax.figure.canvas.mpl_connect("button_press_event", self)
        self.cid_KEYBOARD = self.ax.figure.canvas.mpl_connect("key_press_event", self)


def compute_differential_reddening(
    df: pd.DataFrame,
    params: dict,
    member_threshold: float = 0.9,
    calc_correction_threshold: bool = True,
    roi: Optional[List[Tuple[float, float]] | Polygon] = None,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Polygon]]:
    """
    Computes the differential reddening correction for a cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the photometric data.
    params : dict
        Dictionary of parameters (e.g., column names, extinction coefficients, MS turn-off).
    member_threshold : float, optional
        Membership probability threshold to select reference stars, by default 0.9.
    calc_correction_threshold : bool, optional
        If True, applies a 2-sigma threshold to the evaluated differential reddening, by default True.
    roi : list of tuples or Polygon, optional
        Region of interest in the transformed CMD defining the main sequence, by default None.
        If None, triggers an interactive GUI selection (requires a display).
    plot : bool, optional
        If True, plots the fiducial line fit, by default False.

    Returns
    -------
    band1 : np.ndarray
        Corrected magnitudes for band 1.
    band2 : np.ndarray
        Corrected magnitudes for band 2.
    delta_E_B_V : np.ndarray
        Evaluated differential reddening for each star.
    poli : Polygon or None
        The selected region of interest (Polygon) used for defining MS reference stars.
    """

    # setup E(B-V)
    # exctintion laws
    rband1 = 3.1 * params["rband1"]
    rband2 = 3.1 * params["rband2"]
    A_band1 = rband1
    A_band2 = rband2
    delta_A = A_band1 - A_band2
    theta = np.arctan2(A_band1, delta_A)

    # cluster related params
    turn_off_mag = params["TO_mag"]
    turn_off_color = params["TO_color"]

    # closte neibourghs reference stars number used for correction
    n_ref = params.get("n_reference_stars", params.get("nref", 50))

    # dataframe columns name
    # columns = params["columns"]  # columns name

    x = df[params["xcol"]].values  # x
    x_ref = df.loc[df["membership"] > member_threshold, params["xcol"]].values  # x
    y = df[params["ycol"]].values  # y
    y_ref = df.loc[df["membership"] > member_threshold, params["ycol"]].values  # y
    band1 = df[params["band1"]].values  # calibrated magnitude in band 1 [mag]
    band1_ref = df.loc[df["membership"] > member_threshold, params["band1"]].values
    band2 = df[params["band2"]].values  # calibrated magnitude in band 2 [mag]
    band2_ref = df.loc[df["membership"] > member_threshold, params["band2"]].values
    nstar = df.shape[0]

    delta_E_B_V_init = df["delta_ebv"].values

    band1_ini = band1.copy()
    band2_ini = band2.copy()

    color_obs = band1 - band2
    color_obs_ref = band1_ref - band2_ref
    median_delta_abs = np.zeros(nstar)

    # start process
    # translation (now the orband2gin of the reference frame is the MS TO)
    band1_t = band1 - params["TO_mag"]
    color_obs_t = color_obs - params["TO_color"]

    band1_t_ref = band1_ref - params["TO_mag"]
    color_obs_t_ref = color_obs_ref - params["TO_color"]

    # rotation of the CMD (tilt the CMD along the direction of the reddening vector)
    # in this new reference frame the reddening vector is parallel to the x axis
    # A(lambda)/A(V) extinction coefficients for the two bands
    abscissa = color_obs_t * np.cos(theta) + band1_t * np.sin(theta)  # new X
    ordinate = -color_obs_t * np.sin(theta) + band1_t * np.cos(theta)  # new Y
    abscissa_ref = color_obs_t_ref * np.cos(theta) + band1_t_ref * np.sin(
        theta
    )  # new X
    ordinate_ref = -color_obs_t_ref * np.sin(theta) + band1_t_ref * np.cos(
        theta
    )  # new Y

    if roi is None:
        # spawn cmd widget to choose box
        myROI = ROISelector(np.vstack((abscissa_ref, ordinate_ref)).T)
        myROI.start_selection()  # need to be blocking until selection
        while True:
            if myROI.points_in is not None:
                break
        poli = myROI.polygon
        filter = myROI.points_in
    else:
        poli = Polygon(roi) if not isinstance(roi, Polygon) else roi
        filter = shapely.contains_xy(poli, abscissa_ref, ordinate_ref)

    # _, ax = plt.subplots(layout="constrained")
    # ax.scatter(abscissa, ordinate, marker=".", s=0.5, lw=0.1, zorder=2)
    # plotting.plot_polygon(poli, ax=ax, color="gray", alpha=0.5, zorder=3)
    # plt.show()
    # Define MS stars
    ms_ord = ordinate_ref[filter]
    ms_abs = abscissa_ref[filter]
    ms_x = x_ref[filter]
    ms_y = y_ref[filter]
    band1_ref[filter]

    # here setup the iteration
    ord_step = params["ord_step"] if "ord_step" in params else 0.1
    ord_max = ms_ord.max()
    ord_min = ms_ord.min()
    n_step = (ord_max - ord_min) / ord_step
    n_step = int(round(n_step))

    # Define fiducial line
    # calculate running median in the box and fit on the medians using binned_statistic
    from scipy.stats import binned_statistic

    bins = np.linspace(ord_min, ord_min + ord_step * n_step, n_step + 1)

    median_abs, _, _ = binned_statistic(ms_ord, ms_abs, statistic="median", bins=bins)
    median_ord, _, _ = binned_statistic(ms_ord, ms_ord, statistic="median", bins=bins)

    # Filter out empty bins (where binned_statistic returns NaN)
    valid_bins = ~np.isnan(median_abs) & ~np.isnan(median_ord)
    median_abs = median_abs[valid_bins]
    median_ord = median_ord[valid_bins]

    # faccio il fit
    interp_ord = CubicSpline(median_ord, median_abs)
    fiducial_line_ord_all = interp_ord(ms_ord)
    # interp_abs = CubicSpline(median_abs, median_ord)
    # fiducial_line_abs_all = interp_abs(ms_abs)

    if plot:
        _, ax = plt.subplots(layout="constrained")
        ax.scatter(
            ms_abs,
            ms_ord,
            marker="x",
            s=2,
            lw=0.5,
            color="orange",
            label="reference stars",
            zorder=3,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.scatter(
            abscissa_ref[filter],
            ordinate_ref[filter],
            marker=".",
            s=2,
            lw=1,
            color="black",
            label="all stars",
            zorder=2,
        )
        ax.plot(
            fiducial_line_ord_all[np.argsort(ms_ord)],
            np.sort(ms_ord),
            color="red",
            lw=1,
            label="Fiducial line",
            zorder=3,
        )
        ax.set(xlim=xlim, ylim=ylim, xlabel="abscissa", ylabel="ordinate")
        # ax.set(ylim=(-3, 1))
        ax.invert_xaxis()
        ax.legend()
        # plt.show(block=False)
        plt.draw()
        plt.pause(1.0)

    ########## STEP 5: search for nearest neighbours ##############
    from scipy.spatial import cKDTree

    # define the set of reference coordinates (to build the tree) and the set of points (to look for k neighbours)
    ref_coord = np.column_stack((ms_x, ms_y))
    points = np.column_stack((x, y))

    # KDTree
    tree = cKDTree(ref_coord)

    # number of reference stars
    _, indices = tree.query(points, k=n_ref + 1)

    ########## STEP 6: obtain median delta abscissa in the reference stars ##############
    ref_fiducial_line_ord_all = fiducial_line_ord_all[
        (ms_ord <= ord_max) & (ms_ord >= ord_min)
    ]

    # compute delta abscissa
    delta_abscissa = ms_abs - ref_fiducial_line_ord_all

    # Pre-index delta_abs_all
    delta_abs_all = delta_abscissa[indices[:, 1:]]

    # Implement efficient NumPy-vectorized sigma-clipping & median
    medians = np.median(delta_abs_all, axis=1, keepdims=True)
    stds = np.std(delta_abs_all, axis=1, keepdims=True)

    # Create mask for valid elements
    mask = (delta_abs_all > (medians - 2 * stds)) & (
        delta_abs_all < (medians + 2 * stds)
    )

    # Use np.where to replace invalid elements with NaN, then compute nanmedian
    clipped = np.where(mask, delta_abs_all, np.nan)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_delta_abs = np.nanmedian(clipped, axis=1)

    ########## STEP 8: compute new magnitudes ##############
    # abscissa = abscissa - median_delta_abs
    abscissa_eval = abscissa - median_delta_abs

    color_obs_t = abscissa_eval * np.cos(theta) - ordinate * np.sin(theta)
    band1_t = abscissa_eval * np.sin(theta) + ordinate * np.cos(theta)

    color_obs = color_obs_t + turn_off_color
    band1_eval = band1_t + turn_off_mag
    band2_eval = band1_eval - color_obs

    ########## STEP 9: evaluation of the differential reddening ##############
    if calc_correction_threshold:
        A_band1_star = band1_ini - band1_eval
        A_band2_star = band2_ini - band2_eval
        delta_e_v_band = A_band1_star - A_band2_star
        delta_E_B_V = delta_e_v_band / (rband1 - rband2)
        corr_threshold = 2 * np.std(delta_E_B_V)
        print(f"correction threshold at: {corr_threshold:.3f}")
        band1 = np.where(
            np.logical_and(np.abs(delta_E_B_V) > corr_threshold, delta_E_B_V_init != 0),
            band1_eval,
            band1,
        )
        band2 = np.where(
            np.logical_and(np.abs(delta_E_B_V) > corr_threshold, delta_E_B_V_init != 0),
            band2_eval,
            band2,
        )
        # delta_E_B_V = np.where(np.abs(delta_E_B_V) > corr_threshold, delta_E_B_V, 0.0)
        # I = V - color_obs

    else:
        band1 = band1_eval
        band2 = band2_eval

    ########## STEP 10: compute differential reddening ##############
    A_band1_star = band1_ini - band1
    A_band2_star = band2_ini - band2
    delta_e_v_band = A_band1_star - A_band2_star
    delta_E_B_V = delta_e_v_band / (rband1 - rband2)

    return band1, band2, delta_E_B_V, poli
