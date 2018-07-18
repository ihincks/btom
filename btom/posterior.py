import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt

import btom.bases as btb
import btom.utils as btu

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

    @property
    def purity(self):
        r"""
        The posterior distribution of purity, :math:`\operatorname{Tr}(\rho^2)`.

        :type: ``np.ndarray``
        """
        idxs = (np.s_[:],) + np.diag_indices(self.dim)
        return np.real(np.sum(np.matmul(self.states, self.states)[idxs], axis=-1))

    def fidelity(self, fiducial_state):
        r"""
        The posterior distribution of fidelity,
        :math:`(\operatorname{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2`,
        against the given fiducial state.

        :type: ``np.ndarray``
        """
        if isinstance(fiducial_state, qt.Qobj):
            fiducial_state = fiducial_state.full()
        sq_fs = btu.sqrtm_pos(fiducial_state)[np.newaxis,...]
        tmp = np.matmul(sq_fs, np.matmul(self.states, sq_fs))

        # next we want the trace of the square root of tmp, all squared
        # we sum the square roots of the singular values to acheive this
        return np.sum(np.sqrt(np.linalg.svd(tmp, compute_uv=False)), axis=-1) ** 2

    def plot_purity(self, **kwargs):
        """
        Plots the posterior distribution of the :py:attr:`.purity`.

        :param kwargs: Arguments to pass to :py:func:`btom.utils.kde_plot`.
        """
        btu.plot_kde(self.purity, **kwargs)
        plt.xlim(np.clip(plt.gca().get_xlim(),0,1.01))
        plt.yticks([])
        plt.xlabel(r'Purity Tr$(\rho^2)$')
        plt.ylabel('Posterior density')

    def plot_fidelity(self, fiducial_state, **kwargs):
        """
        Plots the posterior distribution of the :py:attr:`.fidelity` against
        the given fiducial state.

        :param kwargs: Arguments to pass to :py:func:`btom.utils.kde_plot`.
        """
        btu.plot_kde(self.fidelity(fiducial_state), **kwargs)
        plt.xlim(np.clip(plt.gca().get_xlim(),0,1.01))
        plt.yticks([])
        plt.xlabel(r'Fidelity $($Tr$\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2$')
        plt.ylabel('Posterior density')

    def plot_bloch(self, fiducial_state=None, axes=None,
            dist_kwargs=None, est_kwargs=None, fiducial_state_kwargs=None
        ):
        """
        For qubit posteriors, plots the state distribution and the Bayes
        estimate on a Bloch sphere.

        :param fiducial_state: An extra state to plot on the Bloch sphere,
            separate from the bayes estimate and the posterior.
        :param mpl_toolkits.Axes3D axes: The 3D axes to plot on. If ``None``,
            a new 3d axis is created.
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

        fig = plt.gcf()
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

    def plot_basis_expansion(self, basis, fiducial_state=None, fiducial_state_label='Fiducial state'):
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

        :param btom.Basis basis: A basis object to calculate expansion
            coefficients with respect to.
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

    def plot_matrix(self, fiducial_state=None, vector_basis=None, fiducial_state_label='Fiducial state', axes=None):
        r"""
        Plots the posterior as a 3D bar plot in the given basis---if none is
        given, the canonical basis is used. Absolute values of matrix elements
        are shown as bar height, and complex phases are shown as colour.
        The 95% symmetric region of each absolute value is shown by the
        part of the bar with full opacity. If a fiducial state is given,
        is is shown as a dashed line wrapping the bars.

        :param btom.Basis vector_basis: A vector basis to calculate matrix
            components with respect to.
        :param fiducial_state: A single fixed state to be plotted along with
            the posterior. Should be an array the same size as the state, or
            a :py:class:`qutip.Qobj`.
        :param str fiducial_state_label: The label for the ``fiducial_state``
            to use in the plot legend.
        :param mpl_toolkits.Axes3D axes: The 3D axes to plot on. If ``None``,
            a new 3d axis is created.
        """
        ax = plt.gcf().add_subplot(111, projection='3d') if axes is None else axes

        # bases
        b = btb.canonical_basis(self.dim) if vector_basis is None else vector_basis
        bm = b.outer_product()

        # compute bar heights and colors
        if b.ndim > 1 or b.size != self.dim:
            raise ValueError('A 1D vector basis with the same dimension as this posterior\'s states is required.')
        coeffs = bm.expansion(self.states)
        bottom = np.percentile(np.abs(coeffs), 5, axis=1)
        top = np.percentile(np.abs(coeffs), 95, axis=1) - bottom
        arg = np.mean(np.angle(coeffs), axis=1)
        colors = btu.complex_cmap(0.5 * (arg + np.pi) / np.pi)

        # bar locations
        x = np.tile(np.arange(b.n_arrays),b.n_arrays) + 0.75
        y = (np.arange(b.n_arrays)).repeat(b.n_arrays) + 0.75

        # draw lower bars with low opacity
        colors[:,3] = 0.3
        w = 0.5
        idx = 0
        fs = np.abs(bm.expansion(fiducial_state)) if fiducial_state is not None else x
        # draw each bar one at a time to force the zorder to be sane...sigh
        for xval, yval, bval, tval, fval in zip(x, y, bottom, top, fs):
            c = colors[idx,:]
            ax.bar3d([xval], [yval], [0], w, w, [bval], color=c, shade=True)
            c[3] = 1
            ax.bar3d([xval], [yval], [bval], w, w, [tval], color=c, shade=True)
            if fiducial_state is not None:
                lab = fiducial_state_label if idx == 0 else None
                plt.plot(
                        [xval+0.24+w/2, xval+0.24+w/2, xval+0.26-w/2],
                        [yval+0.24-w/2, yval+0.26+w/2, yval+0.26+w/2],
                        [fval]*3,
                        '--', c='k', label=lab, zorder=1e8
                    )
            idx += 1

        if fiducial_state is not None:
            plt.legend()

        # this nonsense simply moves the z axis to the left; get rid of it
        # if it ever breaks matplotlib (https://stackoverflow.com/a/25083379)
        zaxis = ax.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        zaxis.axes._draw_grid = False
        tmp_planes = zaxis._PLANES
        zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                         tmp_planes[0], tmp_planes[1],
                         tmp_planes[4], tmp_planes[5])
        zaxis.axes._draw_grid = draw_grid_old

        # draw the colorbar
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        cb = mpl.colorbar.ColorbarBase(
                cax, cmap=btu.complex_cmap,
                norm=mpl.colors.Normalize(-np.pi, np.pi)
            )
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))

        # set the labels
        _set_labels(ax, ['$' + name + '$' for name in b.names], 'x')
        _set_labels(ax, ['$' + name + '$' for name in b.names], 'y')

        ax.view_init(40, 20)
