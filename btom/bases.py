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

def gell_mann_basis(dims, normalize=False):
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

def pauli_basis(n_qubits=1, normalize=False):
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
    return Basis(arr, names=(npre, nsuf, '', names))


class ArrayList(object):
    """
    Represents a list of arrays with names attached to each one.

    Instances are subscriptable, returning a new :py:class:`ArrayList` with
    the selected elements. Note that this will result in a view of an existing
    array if the indexing is not fancy.

        >>> import btom as bt
        >>> b = bt.ArrayList(np.eye(5), names=[x for x in 'ABCDE'])
        >>> b[:3]       // has A, B, C
        >>> b[[2,4,3]]  // has C, E, D

    :param arrays: A list of :py:class:`qutip.Qobj` objects, or anything
        castable into a numeric :py:class:`np.ndarray` array (with the
        first index indexing over arrays).
    :param names: A list of name strings, one for each of the arrays.
        Optionally, a tuple ``(prefix, suffix, joiner, [names])`` where
        ``names`` is a list of name strings, ``prefix`` is a string that
        prefixes each name, ``suffix`` is a string that suffixes each name,
        and ``joiner`` is a string that joins names when this is the first
        member of :py:meth:`~ArrayList.kron`.
    """
    def __init__(self, arrays, names=None):
        if isinstance(arrays[0], Qobj):
            self._array = np.array([a.full() for a in arrays])
        else:
            self._array = np.array(arrays)

        if names is None:
            self._names = (
                    '', '', '',
                    ['A_{{{}}}'.format(idx) for idx in range(self.n_arrays)]
                )
        else:
            if isinstance(names, tuple):
                self._np, self._ns, self._nj, self._names = names
            else:
                self._np, self._ns, self._nj = '', '', ''
                self._names = names

            if len(self._names) != self.n_arrays:
                raise ValueError((
                    'The number of names ({}) must match the number of '
                    'arrays ({})').format(len(self._names), self.n_arrays)
                )

    @property
    def names(self):
        """
        A list of names, one for each of the arrays.

        :type: ``list``
        """
        return [self._np + name + self._ns for name in self._names]

    @property
    def tex_names(self):
        """
        This array list's :py:attr:`.names` wrapped in dollar signs.

        :type: ``list``
        """
        return ['$' + name + '$' for name in self.names]

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

    def _duplicate(self, arrays=None, names=None):
        """
        Returns a new :py:class:`ArrayList` instance, with the same properties
        as this one, unless otherwise specified by the parameters given.
        """
        arr = self.value if arrays is None else arrays
        names = self._name_tuple() if names is None else names
        return ArrayList(arr, names=names)

    def _name_tuple(self, rep_name=None):
        names = self._names if rep_name is None else rep_name
        return (self._np, self._ns, self._nj, names)

    def __getitem__(self, key):
        arr = self.value[key]
        try:
            names = self.names[key]
        except TypeError:
            names = [self.names[idx] for idx in key]
        return self._duplicate(arrays=arr, names=names)


    def __add__(self, other):
        if isinstance(other, ArrayList):
            arr = self.value + other.value
        if isinstance(other, Qobj):
            arr = self.value + other.full()
        else:
            try:
                arr = self.value + other
            except ValueError:
                arr = self.value + other[np.newaxis,...]
        return ArrayList(arr, names=self._name_tuple())


    def __mul__(self, other):
        if isinstance(other, ArrayList):
            arr = self.value * other.value
        if isinstance(other, Qobj):
            arr = self.value * other.full()
        else:
            try:
                arr = self.value * other
            except ValueError:
                arr = self.value * other[np.newaxis,...]
        return ArrayList(arr, names=self._name_tuple())

    def __div__(self, other):
        if isinstance(other, ArrayList):
            arr = self.value / other.value
        if isinstance(other, Qobj):
            arr = self.value / other.full()
        else:
            try:
                arr = self.value / other
            except ValueError:
                arr = self.value / other[np.newaxis,...]
        return ArrayList(arr, names=self._name_tuple())

    def __truediv__(self, other):
        return self.__div__(other)

    def dot(self, other):
        if isinstance(other, ArrayList):
            arr = np.matmul(self.value, other.value)
        if isinstance(other, Qobj):
            arr = np.dot(self.value, other.full())
        else:
            try:
                arr = np.matmul(self.value, other)
            except ValueError:
                arr = np.matmul(self.value, other[np.newaxis,...])
        return ArrayList(arr, names=self._name_tuple())

    def conj(self):
        """
        Takes the complex conjugate of this array list.

        :returns: A new array list.
        :rtype: :py:class:`ArrayList`
        """
        return self._duplicate(arrays=self._array.conj())

    def transpose(self, axes=None):
        """
        Transposes each of the arrays in this list.

        :param axes: The order to permute the axes in, an iterable with the
            same length as :py:attr:`ndim`. If ``None`` (default) reverses the
            order of the axes of each array, but keeps the order of the arrays
            the same.

        :returns: A new array list.
        :rtype: :py:class:`ArrayList`
        """
        axes = [0] + list(range(self.ndim, 0, -1)) if axes is None else [0] + list(axes)
        return self._duplicate(arrays=self._array.transpose(axes))

    def dagger(self):
        """
        Takes the conjugate transpose of each array in this array list.
        If this is an array list of vectors (``ndim==1``), then each vector
        is made into a row vector of shape ``(1,size)``. If ``ndim>1``, then
        the order of the indices is reversed.

        :returns: A new array list.
        :rtype: :py:class:`ArrayList`
        """
        if self._np == '|' and self._ns[:7] == r'\rangle':
            pre, suf = r'\langle ', '|'
        else:
            pre, suf = self._np, self._ns
        names = (pre, suf, self._nj, self._names)
        if self.ndim == 1:
            arr = self.value.conj()[:,np.newaxis,:]
        else:
            axes = [0] + list(range(self.ndim, 0, -1))
            arr = self.value.conj().transpose(axes)
        return self._duplicate(arrays=arr, names=names)

    def trace(self):
        """
        Returns the trace of each array in this array list.

        :returns: A list of trace values of length :py:attr:`n_arrays`.
        :rtype: ``np.ndarray``
        """
        if self.ndim < 2:
            raise ValueError('1D array lists cannot have their trace taken.')
        if not all([sh == self.shape[0] for sh in self.shape]):
            raise ValueError('Arrays must have all dimensions equal.')
        idxs = np.diag_indices(self.shape[0], self.ndim)
        return np.sum(self.value[(np.s_[:],) + idxs], axis=-1)

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
    def size(self):
        """
        The number of elements in each array of this array list; the product
        of :py:attr:`.shape`

        :type: ``int``
        """
        return int(np.prod(self.shape))

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

    def kron(self, other):
        """
        Returns a new :py:class:`ArrayList` whose elements are all
        pairwaise kronecker products this this :py:class:`ArrayList` and
        the given :py:class:`ArrayList`. Names are given as string
        concatenations.

        :param ArrayList other: The array list to kronecker on.

        :returns: A new :py:class:`ArrayList` as described above.
        :rtype: :py:class:`ArrayList`
        """

        new_arr = np.kron(
                self.value[:,np.newaxis,...], other.value[np.newaxis,...]
            )
        new_arr = new_arr.reshape((-1,) + new_arr.shape[2:])
        names = [tn + self._nj + on for tn in self._names for on in other._names]
        return type(self)(new_arr, names=self._name_tuple(names))

    def flatten(self):
        """
        Flattens each array in this list.

        :returns: This array list, where each array in the list has
            been flattened. Does not return a new instance.
        :rtype: :py:class:`ArrayList`
        """
        self._array = self.flat
        self._np = '|'
        self._ns = r'\rangle'
        return self

    def outer_product(self, other=None):
        r"""
        Returns the outer product between vector array lists
        (i.e. :py:attr:`.ndim` is `1`). If another  array list is not provided,
        the outer product is between this array list and itself.

        If this array list has elements :math:`|1\rangle\ldots|n\rangle` and
        the other array list has elements :math:`|1\rangle\ldots|m\rangle`, then
        the resulting array list has elements :math:`|i\rangle\langle j|` for
        :math:`1\leq i\leq n` and :math:`1\leq j\leq n`.

            >>> import btom as bt
            >>> bt.canonical_basis(2).outer_product()
            >>> bt.pauli_basis().flatten().outer_product()

        :param btom.ArrayList other: The other array list to take the
            outer product with.

        :returns: A new array list with ``self.n_arrays * other.n_arrays``
            members each with shape ``(self.shape[0], other.shape[0])``.
        :rtype: :py:class:`ArrayList`
        """
        if other is None:
            other = self
        if self.ndim > 1 or other.ndim > 1:
            raise ValueError('outer_product is only available for 1D array lists')

        arr = self.value.conj()[:,np.newaxis,:,np.newaxis] * other.value[np.newaxis,:,np.newaxis,:]
        arr = arr.reshape(-1, *arr.shape[-2:])
        pre, suf = r'\langle ', '|' if self._np == '|' and self._ns[:7] == r'\rangle' \
            else (self._np, self._ns)
        names = [on + pre + tn + suf for on, tn in product(other.names, self._names)]
        return self._duplicate(arrays=arr, names=names)

    @property
    def _class_name_(self):
        return type(self).__name__

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
                type_name=self._class_name_
            )
        else:
            return r"""
            <strong>{type_name}:</strong>
                dims=${dims}$,
                names=$\\{{{names}\\}}$
            """.format(
                dims=r"\times".join(map(str, self.shape)),
                names=u",".join(self.names),
                type_name=self._class_name_
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
    """
    def __init__(self, arrays, names=None,
            orthogonal=True, normalize=False
        ):
        super(Basis, self).__init__(arrays, names=names)
        self._sq_norms = np.sum(self.flat.conj() * self.flat, axis=-1)
        self._orthogonal = orthogonal
        if not orthogonal:
            raise NotImplementedError('Non-orthogonal bases are not possible at the moment.')
        if normalize:
            self._array = self._array / np.sqrt(self._sq_norms[(np.s_[:],) + (None,) * self.ndim])
            self.norms = np.ones(self.n_arrays)

        self._normalized = np.allclose(self._sq_norms, 1)

    def _duplicate(self, arrays=None, names=None, orthogonal=None, normalize=False):
        """
        Returns a new :py:class:`ArrayList` instance, with the same properties
        as this one, unless otherwise specified by the parameters given.
        """
        arr = self.value if arrays is None else arrays
        names = self._name_tuple() if names is None else names
        orthogonal = self.orthogonal if orthogonal is None else orthogonal
        return Basis(arr, names=names, orthogonal=orthogonal, normalize=normalize)

    @property
    def orthogonal(self):
        """
        Whether this basis is orthogonal.

        :type: ``bool``
        """
        return self._orthogonal

    @property
    def normalized(self):
        """
        Whether this basis is orthogonal.

        :type: ``bool``
        """
        return self._normalized

    @property
    def _class_name_(self):
        if self.orthogonal and self.normalized:
            cls_name = 'Orthonormal'
        elif self.orthogonal and not self.normalized:
            cls_name = 'Orthogonal'
        elif not self.orthogonal and self.normalized:
            cls_name = 'Normalized'
        else:
            cls_name = ''
        return cls_name + ' ' + type(self).__name__

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
                ) / self._sq_norms[:,np.newaxis]
            elif array.shape == self.shape:
                return np.sum(
                    self.flat.conj() *
                    array.flatten()[np.newaxis,:],
                    axis=-1
                ) / self._sq_norms
            else:
                array = ArrayList(array)
                if array.shape != self.shape:
                    raise ValueError('Input incompatible with this basis.')
                return self.expansion(array)
        else:
            raise NotImplementedError('Expansions not implemented for non-orthogonal basis yet.')
