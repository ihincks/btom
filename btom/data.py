import numpy as np

__all__ = [
    'TomographyData', 'StateTomographyData',
    'BinomialTomographyData', 'PoissonTomographyData'
]

class TomographyData(object):
    """
    Base class for objects that store data from tomography experiments.

    :param int dim: The Hilbert space dimension.
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        """
        The Hilbert space dimension.

        :type: ``int``
        """
        return self._dim

    @property
    def stan_data(self):
        """
        A dictionary object summarizing this dataset which can be used as
        input to a relevant stan sampler; see
        :py:class:`btom.StanTomographySampler`.

        :type: ``int``
        """
        return {
            'D': self.dim
        }

    @classmethod
    def simulate(cls):
        """
        Returns a new :py:class:`TomographyData` instance with simulated data.
        """
        raise NotImplemented(('This particular data structure does not have '
                'a simulation function implemented.'))

class StateTomographyData(TomographyData):
    """
    A base class for objects that store data from quantum state tomography
    experiments. In particular, this base class assumes a list of
    measurement operators each of shape ``(dim,dim)``.

    :param btom.utils.NamedArrayList meas_ops: A list of named measurement
        operators. These names will be used in visualizations when relevant.
    """
    def __init__(self, meas_ops):
        self._meas_ops = meas_ops
        if meas_ops.ndim != 2 or meas_ops.shape[0] != meas_ops.shape[1]:
            raise ValueError('meas_ops must be a list of square matrices')
        super(StateTomographyData, self).__init__(self._meas_ops.shape[0])

    @property
    def meas_ops(self):
        """
        """
        return self._meas_ops

    @property
    def n_meas_ops(self):
        return self.meas_ops.n_arrays

    @property
    def stan_data(self):
        sd= super(StateTomographyData, self).stan_data
        sd.update({
                'm': self.n_meas_ops,
                'M_real': np.real(self.meas_ops.value),
                'M_imag': np.imag(self.meas_ops.value)
            })
        return sd

    @classmethod
    def _measurement_results(cls, true_state, meas_ops):
        """
        Returns the measurement value of the given measurement operators under
        the provided true state.

        :param true_state: The true state of the sytem.
        :param btom.ArrayList meas_ops: A list of measurement operators.

        :returns: A 1D array of size ``meas_ops.n_arrays``.
        :rtype: ``np.ndarray``
        """
        return meas_ops.dagger().dot(true_state).trace()


    @classmethod
    def simulate(cls, true_state, meas_ops):
        """
        Returns a new :py:class:`StateTomographyData` instance with simulated
        data.

        :param true_state: The true state of the sytem, used to simulate
            data with. Can be a any array or :py:class:`qutip.Qobj`.
        :param btom.ArrayList meas_ops: A list of measurement operators.

        :returns: A new data set.
        :rtype: :py:class:StateTomographyData`
        """
        raise NotImplementedError(('This particular data structure does not '
                'have a simulation function implemented.'))


class BinomialTomographyData(StateTomographyData):
    def __init__(self, meas_ops, n_shots, results):
        super(BinomialTomographyData, self).__init__(meas_ops)
        if np.array(n_shots).size == 1:
            self._n_shots = (n_shots * np.ones(self.n_meas_ops)).flatten()
        else:
            self._n_shots = np.array(n_shots).astype(np.int)
            if self._n_shots.ndim != 1 or self._n_shots.size != self.n_meas_ops:
                raise ValueError('n_shots must have the same length as meas_ops')
        self._results = np.array(results).astype(np.int)
        if self._results.ndim != 1 or self._results.size != self.n_meas_ops:
            raise ValueError('results must hav ethe same length as meas_ops')

        self._n_shots = self._n_shots.astype(np.int)
        self._results = self._results.astype(np.int)

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

    @classmethod
    def simulate(cls, true_state, meas_ops, n_shots):
        """
        Returns a new :py:class:`BinomialTomographyData` instance with
        simulated data.

        :param true_state: The true state of the sytem, used to simulate
            data with. Can be a any array or :py:class:`qutip.Qobj`.
        :param btom.ArrayList meas_ops: A list of measurement operators.
        :param n_meas: An integer specifying the number of shots to measure
             each operator for, or, a list of integers, one for each
             measurement operator, specifying how many times to measure each.

        :returns: A new data set.
        :rtype: :py:class:BinomialTomographyData`
        """
        probs = StateTomographyData._measurement_results(true_state, meas_ops)
        if not np.allclose(np.imag(probs), 0):
            raise ValueError(('Some probabilities imaginary; check that'
                'your measurements and state are positive semi-definite and '
                'less than the identity'))
        probs = np.real(probs)
        if np.sum(np.logical_or(probs < 0, probs > 1)) > 0:
            raise ValueError(('Some probabilities were not in [0,1]; check that'
                'your measurements and state are positive semi-definite and '
                'less than the identity'))
        try:
            results = np.random.binomial(n_shots, probs)
        except ValueError as e:
            if 'shape mismatch' in e:
                raise ValueError(('n_shots inconsistent with the number of '
                    'measurement operators'))
        return BinomialTomographyData(meas_ops, n_shots, results)

class PoissonTomographyData(StateTomographyData):
    pass
