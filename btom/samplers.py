import numpy as np
import abc
import btom.utils as btu
import warnings

__all__ = [
    'TomographySampler',
    'StanTomographySampler', 'StanStateSampler',
    'BinomialGinibreStateSampler', 'PoissonGinibreStateSampler'
]

def cartesian_factors_to_states(x, y):
    t = lambda z: z.transpose(0,2,1)
    rho_real = np.matmul(x, t(x)) + np.matmul(y, t(y))
    rho_imag = np.matmul(x, t(y)) - np.matmul(y, t(x))
    tr = np.sum(rho_real[(np.s_[:],) + np.diag_indices(self.dim)], axis=-1)
    return (rho_real + 1j * rho_imag) / tr[:,np.newaxis,np.newaxis]

class TomographySampler(metaclass=abc.ABCMeta):
    """
    Instances of this abstract base class yield samples of a distribution
    over tomography quantities (ie states or processes).
    """
    @abc.abstractmethod
    def sample(self, data):
        """
        Returns samples from a distribution of tomography quantities.

        :param btom.TomographyData data: Tomography data compatible
            with this sampler.
        """
        pass

class StanTomographySampler(TomographySampler):
    """
    A :py:class:`TomographySampler` that draws samples by executing a stan
    program. This class is still abstract as it does not implement the
    :py:meth:`~TomographySampler.sample` method.

    :param StanModelFactory stan_model_factory: The model factor for the stan
        program.
    :param int n_chains: The number of MCMC chains to run when sampling.
    :param int n_iter: The number of iterations per chain. This includes
        burn-in, so only half of this number will be reported per chain.
    :param dict sampling_kwargs: Other named argements to pass to the
        model's sampling method.
    """
    def __init__(self, stan_model_factory, n_chains=3, n_iter=500, sampling_kwargs=None):
        self._n_chains = n_chains
        self._n_iter = n_iter
        self._stan_model_factory = stan_model_factory
        self._sampling_kw_args = {} if sampling_kwargs is None else sampling_kwargs

    @property
    def n_chains(self):
        """
        The number of MCMC chains to run when sampling.

        :rtype: ``int``
        """
        return self._n_chains

    @property
    def n_iter(self):
        """
        The number of iterations per chain. This includes burn-in, so only half
        of this number will be reported per chain.

        :rtype: ``int``
        """
        return self._n_iter

    @property
    def stan_model(self):
        """
        The stan model object of this sampler's model factory.

        :rtype: :py:class:`pystan.StanModel`
        """
        return self._stan_model_factory.model

    def check_data(self, stan_data):
        """
        Checks goodness given ``stan_data`` dictionary relative to this sampler;
        this method is run before sampling, and raises errors if it
        detects problems.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        """
        pass

    def modify_stan_data(self, stan_data):
        """
        Modifies the ``stan_data`` dictionary prior to sampling. This can
        be used to specify parameterizations of the prior distribution,
        for example.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        :returns: An updated ``stan_data`` dictionary that should be valid
             to run on this sampler's stan model.
        :rtype: ``dict``
        """
        return stan_data

    def _raw_sample(self, stan_data):
        """
        Runs the model's sampler and returns the pystan fit object.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        """
        stan_data = self.modify_stan_data(stan_data)
        self.check_data(stan_data)

        with warnings.catch_warnings():
            # getting a dumb future warning to do with np.floating; suppress it
            warnings.simplefilter("ignore")
            fit = self.stan_model.sampling(
                    stan_data,
                    iter=self.n_iter,
                    chains=self.n_chains,
                    **self._sampling_kw_args
                )
        return fit

class StanStateSampler(StanTomographySampler):
    """
    A :py:class:`StanTomographySampler` which samples quantum states
    (density matrices). Note that this assumes the given stan program
    generates quantities with names ``'rho_real'`` and ``'rho_imag'`` that
    report the real and imaginary parts of the state, respectively.

    :param StanModelFactory stan_model_factory: The model factor for the stan
        program.
    :param int n_chains: The number of MCMC chains to run when sampling.
    :param int n_iter: The number of iterations per chain. This includes
        burn-in, so only half of this number will be reported per chain.
    :param dict sampling_kwargs: Other named argements to pass to the
        model's sampling method.
    """
    def sample(self, data):
        """
        Samples quantum states using this sampler's stan program.

        :param btom.TomographyData data: Tomography data compatible
            with this sampler.
        :returns: A tuple ``(states, fit)`` where ``states`` is an
            array of shape ``(n_samples, d, d)`` where ``n_samples``
             is the number of samples, ``d`` is the Hilbert space dimension,
             and the entry at ``[idx,:,:]`` is a density matrix, and
             where ``fit`` is a stan fit object.
        :rtype: (``np.ndarray``, stan fit)
        """
        fit = self._raw_sample(data.stan_data())
        return fit['rho_real'] + 1j * fit['rho_imag'], fit

class BinomialGinibreStateSampler(StanStateSampler):
    r"""
    A :py:class:`StanStateSampler` whose measurements are
    positive operators :math:`\{M_1,\ldots,M_m\}` such that
    :math:`0\leq M_k \leq \mathbb{I}`, using a binomial likelihood as the
    data distribution. The prior distribution is the Ginibre ensemble.

    :param int ginibre_dim: The maximum rank of density operators with
        prior support. If ``None``, the dimension of the Hilbert space will
        be used, so that maximum rank is supported.
    :param int n_chains: The number of MCMC chains to run when sampling.
    :param int n_iter: The number of iterations per chain. This includes
        burn-in, so only half of this number will be reported per chain.
    :param dict sampling_kwargs: Other named argements to pass to the
        model's sampling method.
    """
    def __init__(self, ginibre_dim=None, n_chains=3, n_iter=500, sampling_kwargs=None):
        super(BinomialGinibreStateSampler, self).__init__(
                btu.StanModelFactory.load_builtin('binomial-ginibre.stan'),
                n_chains=n_chains, n_iter=n_iter,
                sampling_kwargs=sampling_kwargs
            )
        self._ginibre_dim = ginibre_dim

    @property
    def ginibre_dim(self):
        """
        The maximum rank of density operators with prior support.

        :type: ``int``
        """
        return self._ginibre_dim

    def check_data(self, stan_data):
        """
        Checks goodness given ``stan_data`` dictionary relative to this sampler;
        this method is run before sampling, and raises errors if it
        detects problems.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        """
        for key in ['D', 'K', 'm', 'M_real', 'M_imag', 'n', 'k']:
            if key not in stan_data:
                raise ValueError(('This stan data does not contain a '
                    'necessary entry for {}').format(key))

    def modify_stan_data(self, stan_data):
        """
        Modifies the ``stan_data`` dictionary prior to sampling by updating it
        with :py:attr:`ginibre_dim`.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        :returns: An updated ``stan_data`` dictionary that should be valid
             to run on this sampler's stan model.
        :rtype: ``dict``
        """
        stan_data = super(BinomialGinibreStateSampler, self).modify_stan_data(stan_data)
        K = stan_data['D'] if self.ginibre_dim is None else self.ginibre_dim
        stan_data['K'] = K
        return stan_data

class PoissonGinibreStateSampler(StanStateSampler):
    r"""
    A :py:class:`StanStateSampler` whose measurements are todo

    :param int ginibre_dim: The maximum rank of density operators with
        prior support. If ``None``, the dimension of the Hilbert space will
        be used, so that maximum rank is supported.
    :param int n_chains: The number of MCMC chains to run when sampling.
    :param int n_iter: The number of iterations per chain. This includes
        burn-in, so only half of this number will be reported per chain.
    :param dict sampling_kwargs: Other named argements to pass to the
        model's sampling method.
    """
    def __init__(self, ginibre_dim=None, dark_flux_est=0, dark_flux_std=0, n_chains=3, n_iter=500, sampling_kwargs=None):
        super(PoissonGinibreStateSampler, self).__init__(
                btu.StanModelFactory.load_builtin('poisson-ginibre.stan'),
                n_chains=n_chains, n_iter=n_iter,
                sampling_kwargs=sampling_kwargs
            )
        self._ginibre_dim = ginibre_dim
        self._dark_flux_est = dark_flux_est
        self._dark_flux_std = dark_flux_std

    @property
    def ginibre_dim(self):
        """
        The maximum rank of density operators with prior support.

        :type: ``int``
        """
        return self._ginibre_dim

    def check_data(self, stan_data):
        """
        Checks goodness given ``stan_data`` dictionary relative to this sampler;
        this method is run before sampling, and raises errors if it
        detects problems.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        """
        for key in ['D', 'K', 'm', 'M_real', 'M_imag', 'counts']:
            if key not in stan_data:
                raise ValueError(('This stan data does not contain a '
                    'necessary entry for {}').format(key))

    def modify_stan_data(self, stan_data):
        """
        Modifies the ``stan_data`` dictionary prior to sampling by updating it
        with :py:attr:`ginibre_dim`.

        :param dict stan_data: The ``data`` argument passed to the stan model's
            sampler.
        :returns: An updated ``stan_data`` dictionary that should be valid
             to run on this sampler's stan model.
        :rtype: ``dict``
        """
        stan_data = super(PoissonGinibreStateSampler, self).modify_stan_data(stan_data)
        K = stan_data['D'] if self.ginibre_dim is None else self.ginibre_dim
        stan_data['K'] = K
        stan_data['dark_flux_est'] = self._dark_flux_est
        stan_data['dark_flux_std'] = self._dark_flux_std
        return stan_data
