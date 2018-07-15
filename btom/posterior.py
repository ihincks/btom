import numpy as np
import matplotlib.pyplot as plt

import btom.bases as btb

__all__ = ['TomographyPosterior', 'StatePosterior']

class TomographyPosterior(object):
    """
    Base class to represent the posterior distribution of a tomography
    experiment.

    :param btom.TomographySampler sampler: The sampler to use, which itself
        contains the prior distribution.
    :param btom.TomographyData data: The dataset to use, must be compatible
        with the sampler.
    """
    def __init__(self, sampler, data):
        self._sampler = sampler
        self._data = data
        self._samples = sampler.sample(data)

    @property
    def data(self):
        return self._data

    @property
    def sampler(self):
        return self._sampler

    @property
    def dim(self):
        return self.data.dim

class StatePosterior(TomographyPosterior):

    @property
    def states(self):
        return self._samples

    def bloch_plot(self, basis=None):
        if basis is None:
            arr = np.zeros((self.dim, 2))
            arr[0,0] = 1; arr[1,1] = 1
            basis = btb.Basis(arr, names=['0', '1'], name_prefix='|', name_suffix=r'\rangle')
        if basis.ndim != 1 or basis.n_arrays != 2 or basis.shape != (self.dim,):
            raise ValueError('The basis must specify a 2-dimensional subspace.')

    def basis_expansion_plot(self, basis):
        """
        Warning: real is called.
        """
        coeffs = np.real(basis.expansion(self.states))
        plt.plot(coeffs)
