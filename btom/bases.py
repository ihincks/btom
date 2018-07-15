import numpy as np
from qutip import Qobj
from itertools import product
from functools import reduce


def _format_float_as_latex(c, tol=1e-10):
    # taken from https://github.com/QInfer/python-qinfer/blob/e90cc57d50f1b48148dbd0c671eff6246dda6c31/src/qinfer/tomography/bases.py
    if abs(c - int(c)) <= tol:
        return str(int(c))
    elif 1e-3 <= abs(c) <= 1e3:
        return u"{:0.3f}".format(c)
    else:
        return (u"{:0.3e}".format(c)).replace("e", r"\times10^{") + "}"


def _format_complex_as_latex(c, tol=1e-10):
    # taken from https://github.com/QInfer/python-qinfer/blob/e90cc57d50f1b48148dbd0c671eff6246dda6c31/src/qinfer/tomography/bases.py
    if abs(c.imag) <= tol:
        # Purely real.
        return _format_float_as_latex(c.real, tol=tol)
    elif abs(c.real) <= tol:
        return _format_float_as_latex(c.imag, tol=tol) + r"\mathrm{i}"
    else:
        return u"{} + {}\mathrm{{i}}".format(
            _format_float_as_latex(c.real, tol=tol),
            _format_float_as_latex(c.imag, tol=tol)
        )

def sparse_mat(dim, *triples):
    mat = np.zeros((dim, dim), dtype=np.complex128)
    for val, idx_row, idx_col in triples:
        mat[idx_row, idx_col] = val
    return mat

def gell_mann_basis(dims, normalize=True):
    """
    Returns a :py:class:`Basis` of the Gell-Mann generalized basis
    for each subsystem, tensored together.

    :param dims: A list of subsystem dimensions, or, an integer for the case
        of a single system.
    :param bool normalize: Whether to trace-normalize the basis.

    :returns: The tensor of Gell-Mann bases for each subsystem.
    :rtype: :py:class:`Basis`
    """
    try:
        if len(dims) > 1:
            b_first = gell_mann_basis(dims[0], normalize=normalize)
            b_rest = gell_mann_basis(dims[1:], normalize=normalize)
            return b_first.kron(b_rest)
        else:
            dim = dims[0]
    except TypeError:
        dim = dims

    diag_names = []
    diag_elems = []
    sym_names = []
    sym_elems = []
    asym_names = []
    asym_elems = []

    for idx_row in range(dim):
        if idx_row > 0:
            diag = [1] * idx_row + [-idx_row] + [0] * (dim - idx_row - 1)
            diag_elems.append(np.diag(diag) /    np.sqrt(idx_row * (idx_row + 1) / 2))
            diag_names.append('Z_{{{}}}'.format(idx_row) if dim > 2 else 'Z')
        for idx_col in range(idx_row+1, dim):
            sym_elems.append(
                    sparse_mat(dim, (1, idx_row, idx_col), (1, idx_col, idx_row))
                )
            sym_names.append('X_{{{},{}}}'.format(idx_row, idx_col) if dim > 2 else 'X')
            asym_elems.append(
                    sparse_mat(dim, (-1j, idx_row, idx_col), (1j, idx_col, idx_row))
                )
            sym_names.append('Y_{{{},{}}}'.format(idx_row, idx_col) if dim > 2 else 'Y')
    return Basis(
            [np.eye(dim)] + sym_elems + asym_elems + diag_elems,
            ['I'] + sym_names + asym_names + diag_names,
            orthogonal=True,
            normalize=normalize
        )

def pauli_basis(n_qubits=1, normalize=True):
    """
    Returns a :py:class:`Basis` of the n-qubit Pauli basis.

    :param int n_qubits: The number of qubits in the system.
    :param bool normalize: Whether to trace-normalize the basis.

    :returns: The tensor of ``n_qubits`` Pauli bases, trace-normalized by
        default.
    :rtype: :py:class:`Basis`
    """
    return gell_mann_basis([2] * n_qubits, normalize=normalize)

def canonical_basis(subsystem_shapes):
    """
    Returns the canonical :py:class:`Basis` of the given shape.

    .. code-block:: python

        import btom as bt
        // canonical basis for qutrit
        b = bt.canonical_basis(3)
        // canonical basis for qutrit tensored with qubit
        b = bt.canonical_basis([3,2])
        // canonical basis for matrices of shape (3,2)
        b = bt.canonical_basis((3,2))
        // canonical basis for matrices of shape (2,2) tensored with qutrit
        b = bt.canonical_basis([(2,2),3])

    .. note:

        The argument to this function makes a strong distiction between
        lists and tuples, the former corresponding to subsystems, and the
        latter corresponding to subsystem shapes.

    :param subsystem_shapes: An integer specifying the dimension of a vector
        basis, or a list of integers specifying the dimensions of each
        vector subsystem. Integers can be replaced by tuples of integers,
        specifying array shapes, for matrix and higher index bases.
    :returns: A basis with elements of the form ``[0,...,0,1,0,...,0]`` for
        vectors, of form ``[[0,...,0],...,[0,...,0,1,0,...,0],...,[0,...,0]]``
        for matrices, and so on for higher dimensional arrays.
    :rtype: :py:class:`Basis`
    """
    ss = subsystem_shapes
    if isinstance(ss, list) and len(ss) > 1:
        return canonical_basis(ss[0]).kron(canonical_basis(ss[1:]))
    elif isinstance(ss, list):
        return canonical_basis(ss[0])

    if not isinstance(ss, tuple):
        ss = (ss,)
    dim = np.prod(ss)

    arr = np.eye(dim).reshape(dim, *ss)
    if len(ss) == 1:
        names = ['{:d}'.format(idx) for idx in range(ss[0])]
        npre, nsuf = r'|', r'\rangle'
    else:
        names = [
                ('E_{{' + '{}'*len(ss) + '}}').format(*idx)
                for idx in product(*[range(d) for d in ss])
            ]
        npre, nsuf = '', ''
    return Basis(arr, names=names, name_suffix=nsuf, name_prefix=npre)


class ArrayList(object):
    """
    Represents a list of arrays with names attached to each one.

    :param arrays: A list of :py:class:`qutip.Qobj` objects, or anything
        castable into a numeric :py:class:`np.ndarray` array (with the
        first index indexing over arrays).
    :param list names: A list of names, one for each of the arrays.
    :param str name_prefix: A string to prepend to each name.
    :param str name_suffix: A string to append to each name.
    """
    def __init__(self, arrays, names = None, name_prefix='', name_suffix=''):
        if isinstance(arrays[0], Qobj):
            self._array = np.array([a.full() for a in arrays])
        else:
            self._array = np.array(arrays)

        if names is None:
            self._names = ['A_{{{}}}'.format(idx) for idx in range(self.n_arrays)]
        else:
            self._names = names
            if len(self._names) != self.n_arrays:
                raise ValueError((
                    'The number of names ({}) must match the number of '
                    'arrays ({})').format(len(names), self.n_arrays)
                )
        self._np = name_prefix
        self._ns = name_suffix

    @property
    def names(self):
        """
        A list of names, one for each of the arrays.

        :type: ``list``
        """
        return [self._np + name + self._ns for name in self._names]

    @property
    def value(self):
        """
        All arrays in this list, as a single ``np.ndarray``, where the first
        index is over elements of the list.

        :type: ``np.ndarray``
        """
        return self._array

    @property
    def flat(self):
        """
        The :py:attr:`value` of this array, but where each array has been
        flattenened, to give the total shape ``(n_arrays, product(shape))``
        for this property.

        :type: ``np.ndarray``
        """
        return self._array.reshape(self.n_arrays, -1)

    def __getitem__(self, *key):
        return self.value.__getitem__(*key)

    @property
    def n_arrays(self):
        """
        The number of arrays in this list.

        :type: ``int``
        """
        return self.value.shape[0]

    @property
    def shape(self):
        """
        The shape of each array in this list.

        :type: ``tuple``
        """
        return self.value.shape[1:]

    @property
    def dtype(self):
        """
        The numpy datatype of this array list.
        """
        return self.value.dtype

    @property
    def ndim(self):
        """
        The number of dimensions of each element of this array list.

        :type: ``int``
        """
        return self.value.ndim - 1

    def kron(self, other, sep_str=''):
        """
        Returns a new :py:class:`ArrayList` whose elements are all
        pairwaise kronecker products this this :py:class:`ArrayList` and
        the given :py:class:`ArrayList`. Names are given as string
        concatenations.

        :param ArrayList other: The array list to kronecker on.
        :param str sep_str: The string to used in between names from each
            array list.

        :returns: A new :py:class:`ArrayList` as described above.
        :rtype: :py:class:`ArrayList`
        """

        new_arr = np.kron(
                self.value[:,np.newaxis,...], other.value[np.newaxis,...]
            )
        new_arr = new_arr.reshape((-1,) + new_arr.shape[2:])
        names = [tn + sep_str + on for tn in self._names for on in other._names]
        return type(self)(new_arr, names, name_prefix=self._np, name_suffix=self._ns)

    def _repr_html_(self):
        # modified from https://github.com/QInfer/python-qinfer/blob/e90cc57d50f1b48148dbd0c671eff6246dda6c31/src/qinfer/tomography/bases.py
        max_shape_allowed = 6 if self.ndim >= 2 else 10
        if max(self.shape) < max_shape_allowed and self.ndim <= 2:
            element_strings = [r"""
                {label} =
                \left(\begin{{matrix}}
                    {rows}
                \end{{matrix}}\right)
                """.format(
                    rows=u"\\\\".join([
                        u"&".join(map(_format_complex_as_latex, row))
                        for row in element
                    ]),
                    label=label
                )
                for element, label in zip(np.atleast_3d(self.value), self.names)
            ]

            return r"""
            <strong>{type_name}:</strong>
                shape=$[{dims}]$
            <p>
                \begin{{equation}}
                    {elements}
                \end{{equation}}
            </p>
            """.format(
                dims=r"\times".join(map(str, self.shape)),
                labels=u",".join(self.names),
                elements=u",".join(element_strings),
                type_name=type(self).__name__
            )
        else:
            return r"""
            <strong>{type_name}:</strong>
                dims=${dims}$,
                names=$\\{{{names}\\}}$
            """.format(
                dims=r"\times".join(map(str, self.shape)),
                names=u",".join(self.names),
                type_name=type(self).__name__
            )

class Basis(ArrayList):
    """
    Represents an ordered basis with names.

    .. note::

        These bases do not need to span the entire space.

    :param arrays: A list of :py:class:`qutip.Qobj` objects, or anything
        castable into a numeric :py:class:`np.ndarray` array (with the
        first index indexing over arrays).
    :param list names: A list of names, one for each of the arrays.
    :param bool orthogonal: Whether this basis is orthogonal.
    :param bool normalize: Whether to normalize the input arrays.
    :param str name_prefix: A string to prepend to each name.
    :param str name_suffix: A string to append to each name.
    """
    def __init__(self, arrays, names,
            orthogonal=True, normalize=False,
            name_prefix='', name_suffix=''
        ):
        super(Basis, self).__init__(
                arrays, names,
                name_prefix=name_prefix, name_suffix=name_suffix
            )
        self.norms = np.sum(self.flat.conj() * self.flat, axis=-1)
        self.orthogonal = orthogonal
        if not orthogonal:
            raise NotImplementedError('Non-orthogonal bases are not possible at the moment.')
        if normalize:
            self._array = self._array / self.norms[(np.s_[:],) + (None,) * self.ndim]
            self.norms = np.ones(self.n_arrays)

    def expansion(self, array):
        """
        Expands the given array (or arrays) in terms of coeffecients with
        respect to this basis.

        :param array: A :py:class:`ArrayList` with arrays of the same shape
            as members of this basis, or an ``np.ndarray`` with the
            same shape as a member of this basis, or an object castable into
            :py:class:`ArrayList` that can then be fed into this expansion.

        :returns: An array of shape ``(self.dims)`` if ``array`` is a single
            array, or an array of shape ``(array.n_arrays, self.dims)`` in the
            case ``array`` is an :py:class:`ArrayList` or something castable
            thereinto.
        :rtype: ``np.ndarray``
        """
        if self.orthogonal:
            if isinstance(array, ArrayList):
                return np.sum(
                    self.flat.conj()[:,np.newaxis,:] *
                    array.flat[np.newaxis,:,:],
                    axis=-1
                ) / self.norms[:,np.newaxis]
            elif array.shape == self.shape:
                return np.sum(
                    self.flat.conj() *
                    array.flatten()[np.newaxis,:],
                    axis=-1
                ) / self.norms
            else:
                array = ArrayList(array)
                if array.shape != self.shape:
                    raise ValueError('Input incompatible with this basis.')
                return self.expansion(array)
        else:
            raise NotImplementedError('Expansions not implemented for non-orthogonal basis yet.')
