# imports
import os
import pandas as pd
import numpy as np

import mgcs_pytools.utils.plotting as utplot
from mgcs_pytools.statistical_membership.membership import compute_statistical_membership

# import the catalog
CATALOG_PATH_CLUSTER = os.path.abspath(...)
CATALOG_PATH_FIELD = os.path.abspath(...)
df_cluster = pd.read_csv(CATALOG_PATH_CLUSTER, ...)
df_field = pd.read_csv(CATALOG_PATH_FIELD, ...)

# setup parameters for the statistical memberhsip
field_fov = ...  # Parallel field FoV in deg^2
cluster_mag = [...]  # Names of the columns with the cluster magnitudes (color, mag)
cluster_mag_error = [
    ...
]  # Names of the columns with the cluster magnitudes (color, mag)
field_mag = [...]  # Names of the columns with the field magnitudes (color, mag)
racol = ...  # Name of the column with the RA
deccol = ...  # Name of the column with the DEC
dr_params = {
    "rband1": ...,  # extinction value in band1
    "rband2": ...,  # extinction value in band2
    "TO_mag": ...,  # Turn-off magnitude
    "TO_color": ...,  # Turn-off color
    "nref": ...,  # number of reference stars
    "ord_step": ...,  # bin size in the 'ordinate' to calculate the ridge line
}

# Region of interest for the reference stars (in the abscissa-ordinate reference frame, not normal CMD)
roi = np.array([[..., ...], [..., ...], [..., ...], [..., ...]])
# if None a matplotlib widget should pop up and you can select the area by clicking the mouse
# !!! BUT THIS DOES NOT WORK ON JUPYTER NOTEBOOKS !!!
# IF YOU ARE ON NOTEBOOK PLEASE PROVIDE THE ROI VALUE MANUALLY AS SHOWN ABOVE
min_field_stars = (
    ...
)  # Minimum number of field stars expected in each annular bin of the target
member_threshold = ...  # Memberhsip threshold for the reference stars in the differential reddening correction
membership_iter = (
    ...
)  # Trial for the statistical subtraction (put high number liker > 100)
min_star_per_cell = (
    ...
)  # Minimum number of stars per Voronoi cell, used for the dilation process

# run the statistical memberhsip
df_output = compute_statistical_membership(
    df_cluster,
    df_field,
    cluster_mag,
    cluster_mag_error,
    field_mag,
    dr_params.copy(),
    parallel_fov=field_fov,
    racol=racol,
    deccol=deccol,
    roi=roi,
    minstars=min_field_stars,
    membership_iter=membership_iter,
    member_threshold=member_threshold,
    min_star_per_cell=min_star_per_cell,
    plot_dred=False,
    plot_voronoi=False,
    which_voronoi="dilation",
    do_dilation=True,
)

# In the df output you will have the membership probabilities in the column "membership"
# as well as the corrected magnitudes for differential reddening in the columns cluster_mag_drcorr
# finally you will find the cumulative delta_ebv in the column "delta_ebv" to plot the differential reddening map


# You can use some of the plotting utilities from the utplot module
fig = utplot.plot_membership_overview(
    btarget=df_output[cluster_mag[0] + "_drcorr"],
    vtarget=df_output[cluster_mag[1] + "_drcorr"],
    bfield=df_field[field_mag[0]],
    vfield=df_field[field_mag[1]],
    membership=df_output["membership"],
)

fig = utplot.plot_decontamination_snapshot(
    df_output[cluster_mag[0] + "_drcorr"],
    df_output[cluster_mag[1] + "_drcorr"],
    df_output[field_mag[0]],
    df_output[field_mag[1]],
    df_output["membership"],
    0.8,
)

fig = utplot.plot_spatial_membership(
    df_output["ra(1)"],
    df_output["dec(2)"],
    df_output["membership_phot"],
    0.9,
)

cluster_center = (
    ...
)  # Something like astropy.coordinates.SkyCoord(ra=..., dec=..., unit="deg")
fig = utplot.plot_differential_reddening_map(
    df_output["ra(1)"],
    df_output["dec(2)"],
    cluster_center.ra.deg,
    cluster_center.dec.deg,
    df_output["delta_ebv"],
)
