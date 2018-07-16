import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt

import btom.bases as btb

__all__ = ['TomographyPosterior', 'StatePosterior']

def _set_axis_xlabels(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

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
        """
        The data used in this posterior.

        :type: :py:class:`btom.TomographyData`
        """
        return self._data

    @property
    def sampler(self):
        """
        The sampler used for this posterior.

        :type: :py:class:`btom.TomographyPosterior`
        """
        return self._sampler

    @property
    def dim(self):
        """
        The Hilbert space dimension of this posterior's sample space.

        :type: ``int``
        """
        return self.data.dim

class StatePosterior(TomographyPosterior):

    @property
    def states(self):
        """
        States (density matrices) sampled from this posterior. This is an
        ``np.ndarray`` of shape ``(n_samples, dim, dim)``.

        :type: ``np.ndarray``
        """
        return self._samples

    @property
    def state_estimate(self):
        """
        The Bayes estimate using this posterior; the mean of :py:attr:`states`.
        This an ``np.ndarray`` of shape ``(dim,dim)``.

        :type: ``np.ndarray``
        """
        return np.mean(self.states, axis=0)

    def bloch_plot(self, fig=None, axes=None, dist_kwargs=None, est_kwargs=None):
        """
        For qubit posteriors, plots the state distribution and the Bayes
        estimate on a Bloch sphere.

        :param matplotlib.pyplot.Figure fig: The matplotlib figure to use. A new
            one is created if none is given.
        :param mpl_toolkits.Axes3D axes: The 3D axes to plot on.
        :param dict dist_kwargs: Arguments to be passed to ``axes.scatter()``
            when plotting the distribution of states.
        :param dict est_kwargs: Arguments to be passed to ``axes.plot()`` when
            plotting the Bayes estimate.

        :returns: The axes object.
        :rtype: ``mpl_toolkits.Axes3D``
        """
        if self.dim != 2:
            raise ValueError('Only available for qubit posteriors.')

        dist_kwargs = {} if dist_kwargs is None else dist_kwargs
        est_kwargs = {} if est_kwargs is None else est_kwargs

        # We have to do some manual jiggering because qutip.Bloch has some
        # limitations in terms of formatting

        if fig is None:
            fig = plt.figure(figsize=(4,4))
        if axes is None:
            axes = Axes3D(fig)
        b = qt.Bloch(fig=fig, axes=axes)
        b.font_size = 12

        axes.set_axis_off()
        b.plot_back()

        paulis = btb.pauli_basis()[1:]
        dist = 2 * np.real(paulis.expansion(self.states)).T
        kwargs = {'edgecolor': None, 'alpha': 0.3, 'marker':'.'}
        kwargs.update(dist_kwargs)
        axes.scatter(dist[:,1], -dist[:,0], dist[:,2], **kwargs)

        b.plot_front()
        b.plot_axes_labels()
        b.plot_axes()

        est = 2 * np.real(paulis.expansion(self.state_estimate))
        kwargs = {'marker':'*', 'color':'r', 'markersize':15, 'markeredgecolor':'pink'}
        kwargs.update(est_kwargs)
        axes.plot([est[1]],[ -est[0]], [est[2]], **kwargs)

        axes.set_xlim3d(-0.7, 0.7)
        axes.set_ylim3d(-0.7, 0.7)
        axes.set_zlim3d(-0.7, 0.7)

        return b.axes

    def basis_expansion_plot(self, basis):
        r"""
        Expands each state in this posterior distribution in terms of the
        given basis, and plots marginal posterior distributions of the
        resulting coefficients.
        Error bars show the Bayes estimates (mean of coefficients), and the
        5% and 95% credible bounds.
        Plots are done on the current matplotlib axis.

        .. note::
            Coefficients are cast to real numbers. Bases do not need to be
            complete.

        :param btom.Basis: A basis object to calculate expansion coefficients
            with respect to.
        """
        coeffs = np.real(basis.expansion(self.states)).T
        plt.violinplot(coeffs,showextrema=False)
        set_axis_xlabels(plt.gca(), basis.names)
        inds = range(1,basis.n_arrays+1)
        ests = np.mean(coeffs, axis=0)
        lims = np.percentile(coeffs, [5, 95], axis=0)
        print(lims.shape)
        lims = [lims[0,:]-ests, ests-lims[1,:]]
        plt.errorbar(inds, ests, c='k', yerr=lims, capsize=5, fmt='D', alpha=0.5)
        #plt.gca().vlines(, lims[:,0], lims[:,1], color='k', linestyle='-', lw=1)
        plt.ylabel(r'Tr$(B_k\rho)$')
        plt.xlabel('Basis elements $B_k$')
        plt.title('Posterior as coefficients of a basis')
