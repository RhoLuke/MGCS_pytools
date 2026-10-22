import numpy as np
from numpy.typing import NDArray
from .bivariate_normal import _logl, _lnprior


def posterior(
    prm: NDArray,
    data: NDArray,
    qmin: NDArray,
    qmax: NDArray,
    component: int,
    pmodel: str = "bivariate_normal",
    fix_params: NDArray | None = None,
    lnk_params: NDArray | None = None,
    stride: int = 5,
) -> float:
    """Posterior function for the MCMC.

    Args:
        prm : `NDArray`
            Parameters to fit.
        data : `NDArray`
            Data to fit.
        qmin : `NDArray`
            Lower limits for priors.
        qmax : `NDArray`
            Upper limits for priors.
        component : `NDArray`
            Number of components to fit.
        pmodel : `str`, optional
            Model to use for the posterior. Default is "bivariate_normal".
            Currently, only "bivariate_normal" is supported.
        fix_params : `NDArray` or `None`, optional
            Array containing the fixed parameters.
            If `None`, no parameters will be fixed. Defaul is `None`
        lnk_params : `NDArray` or `None`, optional
            Array containing the linked parameters.
            If `None`, no parameters will be linked together.
            Defaul is `None`
        stride : `int`, optional
            Number of parameters per component. Default is 5.

    Returns:
        output : `float`
            Logarithm of the posterior probability.
    """

    if pmodel != "bivariate_normal":
        raise ValueError(f"Model {pmodel} is not supported.")

    lp = _lnprior(prm, qmin, qmax, component)

    if fix_params is not None:
        prm_new = np.where(np.isnan(fix_params), prm, fix_params)
    else:
        prm_new = prm.copy()

    if lnk_params is not None:
        for to_link, which in lnk_params:
            prm_new[which] = prm_new[to_link]

    output = (
        -np.inf if not np.isfinite(lp) else lp + _logl(data, prm_new, component, stride)
    )
    return output
