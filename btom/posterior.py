import numpy as np

__all__ = ['TomographyPosterior']

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
        self._states = sampler.sample(data)
