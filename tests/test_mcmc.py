import numpy as np
from mgcs_pytools.mcmc.bivariate_normal import bivariate_normal_pdf, _logl, _lnprior
from mgcs_pytools.mcmc.set_posterior import posterior
from mgcs_pytools.mcmc.mcmc import MCMCEnsembleSampler


def test_bivariate_normal_pdf():
    # Simple check on center (x1=mu1, x2=mu2)
    pdf_val = bivariate_normal_pdf(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    expected = 1.0 / (2.0 * np.pi)
    np.testing.assert_allclose(pdf_val, expected, rtol=1e-5)


def test_logl():
    data = np.array([[0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 0.1, 0.1]])
    prm = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5])
    ll = _logl(data, prm, component=2, stride=5)
    assert not np.isnan(ll)
    assert not np.isinf(ll)


def test_lnprior():
    prm = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    qmin = np.array([-1.0, -1.0, 0.1, 0.1, -0.9, 0.0])
    qmax = np.array([1.0, 1.0, 2.0, 2.0, 0.9, 1.0])

    # Within limits
    val = _lnprior(prm, qmin, qmax, 2)
    assert val == 0.0

    # Outside limits
    prm_bad = np.array([2.0, 0.5, 0.5, 0.5, 0.0, 0.5])
    val_bad = _lnprior(prm_bad, qmin, qmax, 2)
    assert val_bad == -np.inf


def test_mcmc_sampler_synthetic_data():
    np.random.seed(42)
    data = np.random.randn(10, 4)
    data[:, 2:] = 0.1  # error

    # Needs to match ncomp=1 parameters: [mu1, mu2, s1, s2, rho, w] (since mcmc.py defaults to +1 parameters with weights even if component=1)
    # The ndim computation is: ((1 * 5) + 1 = 6 parameters).
    qmin = np.array([-5.0, -5.0, 0.1, 0.1, -0.9, 0.0])
    qmax = np.array([5.0, 5.0, 5.0, 5.0, 0.9, 1.0])

    sampler = MCMCEnsembleSampler(
        data=data,
        nwalkers=12,
        nsteps=5,
        ncomp=1,
        par_per_comp=5,
        qmin=qmin,
        qmax=qmax,
        pmodel="bivariate_normal",
    )

    sampler.run_emcee()
    chain = sampler.get_chain()

    # Shape check: (nsteps, nwalkers, ndim)
    assert chain.shape == (5, 12, 6)


def test_posterior_fixed_linked():
    data = np.array([[0.0, 0.0, 0.1, 0.1]])

    prm = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    qmin = np.array([-1.0, -1.0, 0.1, 0.1, -0.9, 0.0])
    qmax = np.array([1.0, 1.0, 2.0, 2.0, 0.9, 1.0])

    # Test posterior basic
    p_val1 = posterior(prm, data, qmin, qmax, component=1)
    assert not np.isinf(p_val1)

    # Fixed param
    fix_params = np.array([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan])
    p_val2 = posterior(prm, data, qmin, qmax, component=1, fix_params=fix_params)
    assert p_val1 != p_val2

    # Linked param (e.g. sigma2 = sigma1)
    lnk_params = np.array([[2, 3]])
    p_val3 = posterior(prm, data, qmin, qmax, component=1, lnk_params=lnk_params)
    assert p_val1 == p_val3  # because prm has sigma1=1.0 and sigma2=1.0 anyway

    # Change original slightly to see link effect
    prm_diff = np.array([0.0, 0.0, 1.5, 1.0, 0.0, 1.0])
    p_val4 = posterior(prm_diff, data, qmin, qmax, component=1, lnk_params=lnk_params)

    # Now evaluating with prm=[0,0,1.5,1.5,0] vs not linked [0,0,1.5,1.0,0]
    p_val5 = posterior(prm_diff, data, qmin, qmax, component=1)
    assert p_val4 != p_val5
