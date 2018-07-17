import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt

import btom.bases as btb

__all__ = ['TomographyPosterior', 'StatePosterior']

def _set_labels(ax, labels, d='x'):
    if d == 'x':
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
    elif d == 'y':
        ax.get_yaxis().set_tick_params(direction='out')
        ax.yaxis.set_ticks_position('bottom')
        ax.set_yticks(np.arange(1, len(labels) + 1))
        ax.set_yticklabels(labels)
        ax.set_ylim(0.25, len(labels) + 0.75)

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

    def bloch_plot(self, fiducial_state=None, fig=None, axes=None,
            dist_kwargs=None, est_kwargs=None, fiducial_state_kwargs=None
        ):
        """
        For qubit posteriors, plots the state distribution and the Bayes
        estimate on a Bloch sphere.

        :param fiducial_state: An extra state to plot on the Bloch sphere,
            separate from the bayes estimate and the posterior.
        :param matplotlib.pyplot.Figure fig: The matplotlib figure to use. A new
            one is created if none is given.
        :param mpl_toolkits.Axes3D axes: The 3D axes to plot on.
        :param dict dist_kwargs: Arguments to be passed to ``axes.scatter()``
            when plotting the distribution of states.
        :param dict est_kwargs: Arguments to be passed to ``axes.plot()`` when
            plotting the Bayes estimate. Can be used to change label and
            marker, etc.
        :param dict fiducial_state_kwargs: Arguments to be passed to
            ``axes.plot()`` when
            plotting the fiducial state. Can be used to change label and
            marker, etc.

        :returns: The axes object.
        :rtype: ``mpl_toolkits.Axes3D``
        """
        if self.dim != 2:
            raise ValueError('Only available for qubit posteriors.')

        dist_kwargs = {} if dist_kwargs is None else dist_kwargs
        est_kwargs = {} if est_kwargs is None else est_kwargs
        fs_kwargs = {} if fiducial_state_kwargs is None else fiducial_state_kwargs

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
        kwargs = {'edgecolor': None, 'alpha': 0.3, 'marker':'.', 'label':'Posterior sample'}
        kwargs.update(dist_kwargs)
        axes.scatter(dist[:,1], -dist[:,0], dist[:,2], **kwargs)

        b.plot_front()
        b.plot_axes_labels()
        b.plot_axes()

        est = 2 * np.real(paulis.expansion(self.state_estimate))
        kwargs = {'marker':'D', 'color':'k', 'markersize':5, 'label':'Bayes estimate'}
        kwargs.update(est_kwargs)
        axes.plot([est[1]],[ -est[0]], [est[2]], '.', **kwargs)

        if fiducial_state is not None:
            est = 2 * np.real(paulis.expansion(fiducial_state))
            kwargs = {'marker':'*', 'color':'magenta', 'markersize':10, 'label':'Fiducial state'}
            kwargs.update(fs_kwargs)
            axes.plot([est[1]],[ -est[0]], [est[2]], '.', **kwargs)

        axes.set_xlim3d(-0.7, 0.7)
        axes.set_ylim3d(-0.7, 0.7)
        axes.set_zlim3d(-0.7, 0.7)

        plt.legend()
        # plot_axes adds labels for some reason; get rid of
        new_legend = [
                (handle, label)
                for handle, label in zip(*plt.gca().get_legend_handles_labels())
                if label not in 'XYZ'
            ]
        plt.legend([nl[0] for nl in new_legend], [nl[1] for nl in new_legend], loc=4)

        return b.axes

    def basis_expansion_plot(self, basis, fiducial_state=None, fiducial_state_label='Fiducial state'):
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
        :param fiducial_state: A single fixed state to be plotted along with
            the posterior. Should be an array the same size as the state, or
            a :py:class:`qutip.Qobj`.
        :param str fiducial_state_label: The label for the ``fiducial_state``
            to use in the plot legend.
        """
        coeffs = np.real(basis.expansion(self.states)).T
        vp = plt.violinplot(coeffs, showextrema=False)
        _set_labels(plt.gca(), basis.tex_names)
        inds = range(1,basis.n_arrays+1)
        ests = np.mean(coeffs, axis=0)
        lims = np.percentile(coeffs, [5, 95], axis=0)
        print(lims.shape)
        lims = [lims[0,:]-ests, ests-lims[1,:]]
        plt.errorbar(inds, ests, c='k', yerr=lims, capsize=5, fmt='D', alpha=0.5, label=r'90% credible interval')
        #plt.gca().vlines(, lims[:,0], lims[:,1], color='k', linestyle='-', lw=1)

        if fiducial_state is not None:
            for idx, overlap in enumerate(np.real(basis.expansion(fiducial_state))):
                lab = fiducial_state_label if idx == 0 else None
                if idx == 0:
                    fs = plt.plot([idx+0.6, idx+1.4], [overlap, overlap], '--', label=fiducial_state_label)
                    c = fs[0].get_color()
                else:
                    plt.plot([idx+0.6, idx+1.4], [overlap, overlap], '--', color=c)


        plt.ylabel(r'Tr$(B_k\rho)$')
        plt.xlabel('Basis elements $B_k$')
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        vp = mpl.patches.Patch(color=vp['bodies'][0].get_facecolor()[0,:])
        handles.append(vp)
        labels.append('Marginal posterior')
        plt.legend(handles, labels)
