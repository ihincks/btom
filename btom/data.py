import numpy as np

__all__ = ['TomographyData', 'BinomialTomographyData', 'PoissonTomographyData']

class TomographyData(object):
    def __init__(self, meas_ops):
        self._meas_ops = meas_ops
        if meas_ops.n_dim != 2 or meas_ops.shape[0] != meas_ops.shape[1]:
            raise ValueError('meas_ops must be a list of square matrices')

    @property
    def meas_ops(self):
        return self._meas_ops

    @property
    def n_meas_ops(self):
        return self.meas_ops.n_arrays

    @property
    def dim(self):
        return self.meas_ops.shape[0]

    @property
    def stan_data(self):
        return {
            'D': self.dim,
            'm': self.n_meas_ops,
            'M_real': np.real(self.meas_ops),
            'M_imag': np.imag(self.meas_ops)
        }


class BinomialTomographyData(TomographyData):
    def __init__(self, meas_ops, n_shots, results):
        super(BinomialTomographyData, self).__init__(meas_ops)
        if np.array(n_shots).size == 1:
            self._n_shots = (n_shots * np.ones(self.n_meas_ops)).flatten()
        else:
            self._n_shots = np.array(n_shots).astype(np.int)
            if self._n_shots.ndim != 1 or self._n_shots.size != self.n_meas_ops:
                raise ValueError('n_shots must have the same length as meas_ops')
        self._results = np.array(results).astype(np.int)
        if self._results.n_dim != 1 or self._results.size != self.n_meas_ops:
            raise ValueError('results must hav ethe same length as meas_ops')

    @property
    def n_shots(self):
        return self._n_shots

    @property
    def results(self):
        return self._results

    @property
    def stan_data(self):
        sd = super(BinomialTomographyData, self).stan_data
        sd['n'] = self.n_shots
        sd['k'] = self.results
        return sd

class PoissonTomographyData(TomographyData):
    pass
