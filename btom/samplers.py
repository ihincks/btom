import numpy as np
import abc
import btom.utils as btu

__all__ = [
    'TomographySampler', 'StanTomographySampler',
    'BinomialGinibreSampler', 'PoissonGinibreSampler'
]

def cartesian_factors_to_states(x, y):
    t = lambda z: z.transpose(0,2,1)
    rho_real = np.matmul(x, t(x)) + np.matmul(y, t(y))
    rho_imag = np.matmul(x, t(y)) - np.matmul(y, t(x))
    tr = np.sum(rho_real[(np.s_[:],) + np.diag_indices(self.dim)], axis=-1)
    return (rho_real + 1j * rho_imag) / tr[:,np.newaxis,np.newaxis]

class TomographySampler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self):
        pass

class StanTomographySampler(TomographySampler):
    def __init__(self, stan_model, n_chains=3, n_iter=500):
        self.n_chains = 3
        self.n_iter = n_iter
        self.stan_model = stan_model

    def check_data(self, stan_data):
        pass

    def _raw_sample(self, stan_data):
        self.check_data(stan_data)
        fit = self.stan_model(
                stan_data,
                iter=self.n_iter,
                chains=self.n_chains
            )
        return fit.extract()

class BinomialGinibreSampler(StanTomographySampler):
    def __init__(self, ginibre_dim=None, n_chain=3, n_iter=500):
        stan_model = btu.StanModelFactory('').model
        super(BinomialGinibreSampler, self).__init__(
                stan_model,
                n_chains=n_chains, n_iter=n_iter
            )
        self.ginibre_dim = ginibre_dim

    def check_data(self, stan_data):
        for key in ['D', 'K', 'm', 'M_real', 'M_imag', 'n', 'k']:
            if key not in stan_data:
                raise ValueError(('This stan data does not contain the '
                    'necessary entry for {}').format(key))

    def sample(self, stan_data):
        sd = stan_data.copy()
        sd['K'] = sd['D'] if self.ginibre_dim is None else self.ginibre_dim
        params = self._raw_sample(stan_data)
        return cartesian_factors_to_states(
                params['X_real'],
                params['X_imag']
            )

class PoissonGinibreSampler(StanTomographySampler):
    pass
