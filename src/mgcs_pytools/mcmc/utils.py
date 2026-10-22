import numpy as np
from numpy.typing import NDArray


def _bivar_norm_broadcasting_for_chain(
    x1: NDArray,
    x2: NDArray,
    mu1: NDArray,
    mu2: NDArray,
    sigma1: NDArray,
    sigma2: NDArray,
    rho: NDArray,
) -> NDArray:
    """Multivariate normal PDF (2D) with broadcasting for mcmc chain.

    Args:
        x1: `NDArray`
            First variable.
        x2: `NDArray`
            Second variable.
        mu1: `NDArray`
            Mean of the first variable.
        mu2: `NDArray`
            Mean of the second variable.
        sigma1: `NDArray`
            Standard deviation of the first variable.
        sigma2: `NDArray`
            Standard deviation of the second variable.
        rho: `NDArray`
            Correlation coefficient.

    Returns:
        PDF: `NDArray`
            Probability density function values.

    """
    dx1 = x1[:, None] - mu1[None, :]  # (Ndata, Nchain)
    dx2 = x2[:, None] - mu2[None, :]  # (Ndata, Nchain)

    det = sigma1 * sigma2 * np.sqrt(1 - rho**2)
    norm = 1.0 / (2.0 * np.pi * det)

    z = (
        (dx1 / sigma1) ** 2
        + (dx2 / sigma2) ** 2
        - 2 * rho * dx1 * dx2 / (sigma1 * sigma2)
    ) / (2 * (1 - rho**2))

    return norm * np.exp(-z)


def _generate_weighted_mvar_pdf(
    ch: NDArray, x1: NDArray, x2: NDArray, x1e: NDArray, x2e: NDArray
) -> NDArray:
    """Weighted bivariate normal PDF.

    Args:
        ch: `NDArray`
            Chain parameters (Nchain, 5).
        x1: `NDArray`
            First observable (Ndata,).
        x2: `NDArray`
            Second observable (Ndata,).
        x1e: `NDArray`
            First observable error (Ndata,).
        x2e: `NDArray`
            Second observable error (Ndata,).

    Returns:
        `NDArray`
            Weighted bivariate normal PDF (Ndata, Nchain).
    """

    mu1, mu2 = ch[:, 0], ch[:, 1]  # (Nchain,)
    s1, s2 = ch[:, 2], ch[:, 3]  # (Nchain,)
    rho = ch[:, 4]  # (Nchain,)
    w = ch[:, -1]  # (Nchain,)

    sigma1_obs = np.sqrt(s1[None, :] ** 2 + x1e[:, None] ** 2)  # (Ndata,Nchain)
    sigma2_obs = np.sqrt(s2[None, :] ** 2 + x2e[:, None] ** 2)

    # rho_eff
    rho_eff = rho[None, :] * (s1[None, :] * s2[None, :]) / (sigma1_obs * sigma2_obs)

    pdf = _bivar_norm_broadcasting_for_chain(
        x1, x2, mu1, mu2, sigma1_obs, sigma2_obs, rho_eff
    )

    return w[None, :] * pdf  # shape (Ndata, Nchain)


def get_component_membership_from_mcmc_chain(
    ch: NDArray, data: NDArray, ncomp: int, par_per_comp: int, batch_size: int = 1000
) -> NDArray:
    """Get component membership from MCMC chain.

    Parameters
    ----------
    ch : NDArray
        Chain parameters.
    data : NDArray
        Observational data.
    ncomp : int
        Number of components.
    par_per_comp : int
        Number of parameters per component.
    batch_size : int, optional
        Batch size to process chain iterations to avoid OOM, by default 1000.

    Returns
    -------
    NDArray
        Component membership.
    """
    x1, x2, x1e, x2e = data.T  # (Ndata,)
    n_chain = ch.shape[0]

    membership_sum = np.zeros((ncomp, x1.shape[0]))

    for batch_start in range(0, n_chain, batch_size):
        batch_end = min(batch_start + batch_size, n_chain)
        batch_ch = ch[batch_start:batch_end]

        if ncomp > 1:
            w = batch_ch[:, -(ncomp - 1) :]
        else:
            w = 1.0

        pdf = np.empty((ncomp, x1.shape[0], batch_ch.shape[0]))
        for i in range(ncomp):
            c_ch = np.empty((batch_ch.shape[0], par_per_comp + 1))
            c_ch[:, :par_per_comp] = batch_ch[
                :, (i * par_per_comp) : (1 + i) * par_per_comp
            ]

            if ncomp > 1 and i < ncomp - 1:
                c_ch[:, -1] = w[:, i]
            elif ncomp > 1 and i == ncomp - 1:
                c_ch[:, -1] = 1 - np.sum(w, axis=1)
            elif ncomp == 1:
                c_ch[:, -1] = w

            pdf[i] = _generate_weighted_mvar_pdf(
                c_ch, x1, x2, x1e, x2e
            )  # (Ndata, Nchain)

        denom = pdf.sum(axis=0)
        membership_sum += np.sum(pdf / denom[None, :], axis=2)

    membership = membership_sum / n_chain
    return membership
