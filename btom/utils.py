import os
import pystan as ps
import dill

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
BUILTIN_MODEL_DIR = os.path.join(PACKAGE_DIR, "stan")

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
