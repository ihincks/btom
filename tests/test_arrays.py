import btom.utils as btu
import qutip as qt
import numpy as np
import pytest

class TestArrayList(object):
    @pytest.mark.parametrize('n_arrays', [3])
    @pytest.mark.parametrize('shape', [(2,), (3,4)])
    def test_properties(self, n_arrays, shape):
        arr = np.empty([n_arrays] + list(shape))
        al = btu.ArrayList(arr)

        assert np.allclose(arr, al.value)
        assert np.allclose(arr[1], al[1])
        assert np.allclose(arr[1,:], al[1,:])
        assert np.allclose(al.flat, arr.reshape(n_arrays, -1))
        assert al.n_arrays == n_arrays
        assert al.shape == shape
        assert al.dtype == al.dtype
        assert al.ndim == len(shape)

    def test_qobj(self):
        qobj_list = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        al = btu.ArrayList(qobj_list)

        assert np.allclose(al.value, np.array([a.full() for a in qobj_list]))
        assert al.n_arrays == 3
        assert al.ndim == 2
        assert al.shape == (2,2)
        assert al.dtype == np.complex128

class TestNamedArrayList(object):
    @pytest.mark.parametrize('n_arrays,names',
        [(3, ['X', 'Y', 'Z'])]
    )
    @pytest.mark.parametrize('shape', [(2,), (3,4)])
    def test_properties(self, n_arrays, shape, names):
        arr = np.empty([n_arrays] + list(shape))
        al = btu.NamedArrayList(arr, names)

        assert al.names == names
        assert np.allclose(arr, al.value)
        assert np.allclose(arr[1], al[1])
        assert np.allclose(arr[1,:], al[1,:])
        assert np.allclose(al.flat, arr.reshape(n_arrays, -1))
        assert al.n_arrays == n_arrays
        assert al.shape == shape
        assert al.dtype == al.dtype
        assert al.ndim == len(shape)

class TestBasis(object):

    def test_expansion(self):
        qobj_list = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        al = btu.Basis(qobj_list, ['I', 'X', 'Y', 'Z'])

        assert np.allclose(al.expansion(al), np.eye(4))

        x = qt.identity(2).full() + 3 * qt.sigmax().full()
        assert np.allclose(al.expansion(x), [1,3,0,0])

        
