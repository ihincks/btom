import os
import pystan as ps
import dill
import numpy as np
from qutip import Qobj

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
BUILTIN_MODEL_DIR = os.path.join(PACKAGE_DIR, "stan")

def gell_mann_basis(subsystem_dims):


def pauli_basis(n_qubits=2):
    return gell_mann_basis([2] * n_qubits)

class ArrayList(object):
    """
    Represents a list of arrays.
    """
    def __init__(self, arrays, names = None):
        if isinstance(arrays[0], Qobj):
            self._array = np.array([a.full() for a in arrays])
        else:
            self._array = np.array(arrays)

        if names is None:
            self._names = ['{}'.format(idx) for idx in range(self.n_arrays)]
        else:
            self._names = names
            if len(self._names) != self.n_arrays:
                raise ValueError((
                    'The number of names ({}) must match the number of '
                    'arrays ({})').format(len(names), self.n_arrays)
                )

    @property
    def names(self):
        return self._names

    @property
    def value(self):
        return self._array

    @property
    def flat(self):
        return self._array.reshape(self.n_arrays, -1)

    def __getitem__(self, *key):
        return self.value.__getitem__(*key)

    @property
    def n_arrays(self):
        return self.value.shape[0]

    @property
    def shape(self):
        return self.value.shape[1:]

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def ndim(self):
        return self.value.ndim - 1

    def apply(self, fun, *args, **kwargs):
        test = fun(self[0], *args, **kwargs)
        if isinstance(test, np.ndarray):
            pass

class Basis(ArrayList):
    def __init__(self, arrays, names, orthogonal=True, normalize=False):
        super(Basis, self).__init__(arrays, names)
        if self.ndim != 2:
            raise ValueError('OperatorBasis must be an ArrayList of rectangular matrices')
        self.norms = np.sum(self.flat.conj() * self.flat, axis=-1)
        self.orthogonal = orthogonal
        if normalize:
            self._array = self._array / self.norms
            self.norms = 1

    def expansion(self, array):
        if self.orthogonal:
            if isinstance(array, ArrayList):
                return np.sum(
                    self.flat.conj()[:,np.newaxis,:] *
                    array.flat[np.newaxis,:,:],
                    axis=-1
                ) / self.norms
            elif array.shape == self.shape:
                return np.sum(
                    self.flat.conj() *
                    array.flatten()[np.newaxis,:],
                    axis=-1
                ) / self.norms
            elif array.ndim == self.ndim+1 and array.shape[1:] == self.shape:
                return np.sum(
                    self.flat.conj()[:,np.newaxis,:] *
                    array.reshape(array.shape[0],-1)[np.newaxis,:,:],
                    axis=-1
                ) / self.norms
            else:
                raise ValueError('Input incompatible with this basis.')
        else:
            raise NotImplementedError('Expansions not implemented for non-orthogonal basis yet.')

class StanModelFactory(object):
    """
    Class to construct instances of :py:class:`pystan.StanModel`, which first
    checks if the model has already been compiled and saved to disk, loading it
    if it has. Models are saved to disk through pickling, using the
    extension `.pkl`, and stored in the same folder as the given
    text stan program, which usually has the extension `.stan`.

    :param str filename: Filename of stan code to load, including path and
    extension. The pickled model will be stored in the same folder.
    """
    def __init__(self, filename):
        storage_folder = os.path.dirname(os.path.abspath(filename))
        filename = os.path.basename(filename)
        self.stan_filename = os.path.join(storage_folder, filename)
        self.storage_filename = os.path.join(
                storage_folder,
                os.path.splitext(filename)[0] + '.pkl'
            )
        self._model = None

    def _load_model_from_disk(self):
        """
        Tries to load a pickled StanModel object from storage_filename. Returns
        this, or None if it fails.
        """
        try:
            with open(self.storage_filename, 'rb') as f:
                model = dill.load(f)
        except IOError:
            model = None
        return model


    def _save_model_to_disk(self, model):
        """
        Pickles the given model and saves it to file.

        :param model: StanModel object to pickle and save to storage_filename
        """
        with open(self.storage_filename, 'wb') as f:
            dill.dump(model, f)

    def _get_model_code(self):
        """
        Reads _filename and returns its contents as a string.
        """
        with open(self.stan_filename, 'r') as f:
            model_code = "".join(f.read())
        return model_code

    def _up_to_date(self):
        """
        Decides if _model is up-to-date. Returns True if _model exists and
        has model_code equal to the current contents of _filename, False
        otherwise.
        """
        if self._model is None:
            return False
        else:
            return self._model.model_code == self._get_model_code()

    def _get_model(self):
        """
        Loads and unpickles the StanModel from disk if it exists, and returns
        it if it is up-to-date. Otherwise, compiles a new StanModel.
        """
        model = self._load_model_from_disk()
        if model is not None and model.model_code == self._get_model_code():
            return model

        model = ps.StanModel(self.stan_filename)
        self._save_model_to_disk(model)
        return model

    @property
    def model(self):
        """
        A StanModel instance of the model code located at the filename given
        at construction, but up to date with the current contents of the file.
        """
        if not self._up_to_date():
            self._model = self._get_model()
        return self._model

    @classmethod
    def load_builtin(cls, filename):
        return StanModelFactory(os.path.join(BUILTIN_MODEL_DIR, filename))
