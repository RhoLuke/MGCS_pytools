import numpy as np
from numpy.typing import NDArray


def mvar_norm(
    x1: NDArray, x2: NDArray, mu1: NDArray, mu2: NDArray, cov: NDArray
) -> NDArray:
    """Multivariate normal distribution function.

    Parameters
    ----------
    x1 : `NDArray`
        Input along first dimension.
    x2 : `NDArray`
        Input along second dimension.
    mu1 : `NDArray`
        Mean value along first dimension.
    mu2 : `NDArray`
        Mean value along first dimension.
    cov : `NDArray`
        Covariance matrix.

    Returns
    -------
    `NDArray`
        Probability density function.
    """

    diff = np.array([x1 - mu1, x2 - mu2]).T
    minv = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))
    prod1 = np.matmul(minv, diff[:, :, np.newaxis])
    prod2 = np.matmul(diff[:, np.newaxis, :], prod1)

    return norm * np.exp(-0.5 * prod2.T)


def _logl(data, prm, component, stride=5):

    #   unpack data
    pm_ra = data[:, 0]
    pm_dec = data[:, 1]
    epm_ra = data[:, 2]
    epm_dec = data[:, 3]

    if component > 1:
        w = prm[-(component - 1) :]
    else:
        w = prm[-1]

    # create a model for each component
    c_dist = np.zeros(pm_ra.shape)
    for i in range(component):
        cprm = prm[(i * stride) : (i + 1) * stride]
        sigma_ra = np.sqrt(cprm[2] * cprm[2] + epm_ra * epm_ra)
        sigma_dec = np.sqrt(cprm[3] * cprm[3] + epm_dec * epm_dec)
        rho_eff = [
            (cprm[-1] * cprm[2] * cprm[3]) / (c_si_ra * c_si_dec)
            for c_si_ra, c_si_dec in zip(sigma_ra, sigma_dec)
        ]
        cov = np.array(
            [
                np.array(
                    [
                        [c_si_ra * c_si_ra, reff * c_si_ra * c_si_dec],
                        [reff * c_si_ra * c_si_dec, c_si_dec * c_si_dec],
                    ]
                )
                for c_si_ra, c_si_dec, reff in zip(sigma_ra, sigma_dec, rho_eff)
            ]
        )

        if component > 1 and i < component - 1:
            c_dist += w[i] * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[0, 0, :]
        elif component == 1:
            c_dist += w * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[0, 0, :]
        else:
            c_dist += (1 - w.sum()) * mvar_norm(pm_ra, pm_dec, cprm[0], cprm[1], cov)[
                0, 0, :
            ]

    if np.any(np.isnan(np.log(c_dist))):
        print(c_dist)
        print("crashing parameters:", prm)
        return np.nan

    return np.log(c_dist).sum()


def _lnprior(prm, qmin, qmax, ncomp) -> float:
    if ncomp > 1:
        ws = prm[-(ncomp - 1) :]
        if (prm < qmin).any() or (prm > qmax).any() or 1 - ws.sum() < 0.0:
            return -np.inf
        else:
            return 0.0
    else:
        if (prm < qmin).any() or (prm > qmax).any():
            return -np.inf
        else:
            return 0.0
