import numpy as np
import btom.bases as btb

__all__ = [
    "TomographyData",
    "StateTomographyData",
    "BinomialTomographyData",
    "PoissonTomographyData",
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

    def stan_data(self, include_est=False):
        """
        A dictionary object summarizing this dataset which can be used as
        input to a relevant stan sampler; see
        :py:class:`btom.StanTomographySampler`.

        :type: ``int``
        """
        return {"D": self.dim}

    @classmethod
    def simulate(cls):
        """
        Returns a new :py:class:`TomographyData` instance with simulated data.
        """
        raise NotImplemented(
            (
                "This particular data structure does not have "
                "a simulation function implemented."
            )
        )


class StateTomographyData(TomographyData):
    """
    A base class for objects that store data from quantum state tomography
    experiments. In particular, this base class assumes a list of
    measurement operators each of shape ``(dim,dim)``.

    :param btom.ArrayList meas_ops: A list of named measurement operators.
    """

    def __init__(self, meas_ops):
        self._meas_ops = meas_ops
        if meas_ops.ndim != 2 or meas_ops.shape[0] != meas_ops.shape[1]:
            raise ValueError("meas_ops must be a list of square matrices")
        super(StateTomographyData, self).__init__(self._meas_ops.shape[0])

    @property
    def meas_ops(self):
        """
        """
        return self._meas_ops

    @property
    def n_meas_ops(self):
        return self.meas_ops.n_arrays

    def ls_estimate(self):
        """
        Returns the least-squares estimate of the density matrix, working in
        the gell-mann basis and assiming a trace-1 state.

        :returns: Array of shape ``(dim, dim)``.
        :rtype: ``np.ndarray``
        """
        return self.ls_bs_estimates(n_bs=0)[0]

    def ls_bs_estimates(self, n_bs=500):
        """
        Returns the least-squares estimate of the density matrix (see
        :py:meth:`.ls_estimate`), along with a bunch of similar estimates
        obtained using the same estimator for bootstrapped data sets.

        :param int n_bs: The number of bootstrap samples.
        :returns: Tuple ``(est, bs_ests)`` where ``est`` is as in
            :py:meth:`.ls_estimate`, and ``bs_ests`` has shape
            ``(n_bs,dim,dim)`` where the first index is over bootstrap samples.
        """
        raise NotImplementedError(
            "This data type has not " "implemented a least-squares fitter."
        )

    def stan_data(self, include_est=False):
        sd = super(StateTomographyData, self).stan_data(include_est=include_est)
        sd["m"] = self.n_meas_ops
        sd["M_real"] = np.real(self.meas_ops.value)
        sd["M_imag"] = np.imag(self.meas_ops.value)
        if include_est:
            try:
                sd["rho_est"], sd["rho_bs"] = self.ls_bs_estimates()
            except NotImplementedError:
                pass
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
        raise NotImplementedError(
            (
                "This particular data structure does not "
                "have a simulation function implemented."
            )
        )


class BinomialTomographyData(StateTomographyData):
    r"""
    A :py:class:`StateTomographyData` dataset which resulted from summing the
    results of binary measurements. Specifically, this dataset assumes a list
    of :math:`I` two outcome POVMS :math:`\{M_i,\mathbb{I}-M_i\}`, where for
    each :math:`1\leq i \leq I` we have operator inequalities
    :math:`0\leq M_i \leq \mathbb{I}`. Typically :math:`\{M_i\}_{i=1}^I`
    should form a spanning set for density matrices---tensors Pauli matrices
    (scaled and with a added identities to make them positive) are a popular
    choice.

    The :math:`i^\text{th}` POVM is measured :math:`n_i` times, resulting
    in :math:`k_i` results of `1`, which has probability given by
    :math:`Pr(k_i|n_i,M_i,\rho)={n_i \choose k_i}p_i^{k_i}(1-p_i)^{n_i-k_i}`
    where :math:`p_i=\operatorname{Tr}(M_i\rho)`.

    :param btom.ArrayList meas_ops: A list of named measurement
        operators corresponding to :math:`\{M_i\}_{i=1}^I` defined above.
    :param n_shots: An integer or array-like of length :math:`I` corresponding
        to the :math:`n_i` defined above. If an integer, it is assumed that
        all measurement operators are measured the same number of times.
    :param results: An array-like of length :math:`I` corresponding
        to the :math:`k_i` defined above.
    """

    def __init__(self, meas_ops, n_shots, results):
        super(BinomialTomographyData, self).__init__(meas_ops)
        if np.array(n_shots).size == 1:
            self._n_shots = (n_shots * np.ones(self.n_meas_ops)).flatten()
        else:
            self._n_shots = np.array(n_shots)
            if self._n_shots.ndim != 1 or self._n_shots.size != self.n_meas_ops:
                raise ValueError("n_shots must have the same length as meas_ops")
        self._results = np.array(results)
        if self._results.ndim != 1 or self._results.size != self.n_meas_ops:
            raise ValueError("results must hav ethe same length as meas_ops")

        self._n_shots = self._n_shots.astype(np.int)
        self._results = self._results.astype(np.int)

    @property
    def n_shots(self):
        return self._n_shots

    @property
    def results(self):
        return self._results

    def ls_bs_estimates(self, n_bs=500):
        # expand each measurement op in terms of orthonormal hermitian basis
        basis = btb.gell_mann_basis(self.dim, normalize=True)
        X = np.real(basis.expand(self.meas_ops).T)

        # estimate of each measurement's overlap with the state
        # hedge a bit to avoid 0 variance
        y = (self.results + 0.5) / (self.n_shots + 1)
        # draw bootstrap samples
        results_bs = np.random.binomial(self.n_shots, y, size=(n_bs, self.n_meas_ops))
        y_bs = (results_bs + 0.5) / (self.n_shots + 1)

        # we assume unit trace density matrices, thus we know coeff on this
        # basis element is 1/sqrt(d). thus, subtract it off the RHS to
        # avoid making it a fit parameter.
        y -= X[:, 0] / np.sqrt(self.dim)
        y_bs -= X[:, 0] / np.sqrt(self.dim)
        X = X[:, 1:]

        # for some reason this is faster than np.linalg.pinv for me
        Xinv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)

        # return estimate of state in standard basis
        mats_bs = basis[1:].construct(np.dot(y_bs, Xinv)) + np.eye(self.dim) / self.dim
        mat = basis[1:].construct(np.dot(Xinv, y)) + np.eye(self.dim) / self.dim

        return mat, mats_bs

    def stan_data(self, include_est=False):
        sd = super(BinomialTomographyData, self).stan_data(include_est=include_est)
        sd["n"] = self.n_shots
        sd["k"] = self.results
        return sd

    @classmethod
    def simulate(cls, true_state, meas_ops, n_shots):
        """
        Returns a new :py:class:`BinomialTomographyData` instance with
        simulated data.

        :param true_state: The true state of the sytem, used to simulate
            data with. Can be a any array or :py:class:`qutip.Qobj`.
        :param btom.ArrayList meas_ops: A list of measurement operators, see
            :py:class:`BinomialTomographyData` for specification.
        :param n_meas: An integer specifying the number of shots to measure
             each operator for, or, a list of integers, one for each
             measurement operator, specifying how many times to measure each.

        :returns: A new data set.
        :rtype: :py:class:BinomialTomographyData`
        """
        probs = StateTomographyData._measurement_results(true_state, meas_ops)
        if not np.allclose(np.imag(probs), 0):
            raise ValueError(
                (
                    "Some probabilities imaginary; check that"
                    "your measurements and state are positive semi-definite and "
                    "less than the identity"
                )
            )
        probs = np.real(probs)
        if np.sum(np.logical_or(probs < 0, probs > 1)) > 0:
            raise ValueError(
                (
                    "Some probabilities were not in [0,1]; check that"
                    "your measurements and state are positive semi-definite and "
                    "less than the identity"
                )
            )
        try:
            results = np.random.binomial(n_shots, probs)
        except ValueError as e:
            if "shape mismatch" in e:
                raise ValueError(
                    ("n_shots inconsistent with the number of " "measurement operators")
                )
        return BinomialTomographyData(meas_ops, n_shots, results)


class PoissonTomographyData(StateTomographyData):
    r"""
    A :py:class:`StateTomographyData` dataset which resulted from poisson
    counts, such as in quantum optics measurements.
    Specifically, this dataset assumes a list
    of :math:`I` POVMS :math:`\{M_{i,1},\mathbb{I}-M_i\}`, where for
    each :math:`1\leq i \leq I` we have operator inequalities
    :math:`0\leq M_i \leq \mathbb{I}`. Typically :math:`\{M_i\}_{i=1}^I`
    should form a spanning set for density matrices---tensors Pauli matrices
    (scaled and with a added identities to make them positive) are a popular
    choice.

    The :math:`i^\text{th}` POVM is measured :math:`n_i` times, resulting
    in :math:`k_i` results of `1`, which has probability given by
    :math:`Pr(k_i|n_i,M_i,\rho)={n_i \choose k_i}p_i^{k_i}(1-p_i)^{n_i-k_i}`
    where :math:`p_i=\operatorname{Tr}(M_i\rho)`.

    :param btom.utils.NamedArrayList meas_ops: A list of named measurement
        operators corresponding to :math:`\{M_i\}_{i=1}^I` defined above.
    :param n_shots: An integer or array-like of length :math:`I` corresponding
        to the :math:`n_i` defined above. If an integer, it is assumed that
        all measurement operators are measured the same number of times.
    """

    def __init__(self, meas_ops, counts, groups=None):
        super(PoissonTomographyData, self).__init__(meas_ops)
        self._counts = np.array(counts)
        if self.counts.ndim != 1 or self.counts.size != self.n_meas_ops:
            raise ValueError("Counts must have the same length as meas_ops")
            # put everything in one group
        self._groups = (
            np.arange(self.n_meas_ops)[np.newaxis, :]
            if groups is None
            else np.array(groups)
        )

    @property
    def counts(self):
        return self._counts

    @property
    def groups(self):
        return self._groups

    @property
    def n_groups(self):
        return self.groups.shape[0]

    def ls_bs_estimates(self, n_bs=500):
        super(PoissonTomographyData, self).ls_bs_estimates()

    def stan_data(self, include_est=False):
        sd = super(PoissonTomographyData, self).stan_data(include_est=include_est)
        new_order = self.groups.flatten()
        sd["M_real"] = sd["M_real"][new_order, ...]
        sd["M_imag"] = sd["M_imag"][new_order, ...]
        sd["counts"] = self.counts[new_order]
        sd["n_groups"] = self.n_groups
        sd["group_size"] = [group.size for group in self.groups]
        return sd

    @classmethod
    def simulate(cls, true_state, meas_ops, rate):
        """
        Returns a new :py:class:`PoissonTomographyData` instance with
        simulated data.

        :param true_state: The true state of the sytem, used to simulate
            data with. Can be a any array or :py:class:`qutip.Qobj`.
        :param btom.ArrayList meas_ops: A list of measurement operators.
        :param float rate: A number greater than zero representing the total
            flux of counts.

        :returns: A new data set.
        :rtype: :py:class:PoissonTomographyData`
        """
        probs = StateTomographyData._measurement_results(true_state, meas_ops)
        if not np.allclose(np.imag(probs), 0):
            raise ValueError(
                (
                    "Some probabilities imaginary; check that"
                    "your measurements and state are positive semi-definite and "
                    "less than the identity"
                )
            )
        probs = np.real(probs)
        if np.sum(np.logical_or(probs < 0, probs > 1)) > 0:
            raise ValueError(
                (
                    "Some probabilities were not in [0,1]; check that"
                    "your measurements and state are positive semi-definite and "
                    "less than the identity"
                )
            )
        try:
            results = np.random.binomial(n_shots, probs)
        except ValueError as e:
            if "shape mismatch" in e:
                raise ValueError(
                    ("n_shots inconsistent with the number of " "measurement operators")
                )
        return BinomialTomographyData(meas_ops, n_shots, results)
