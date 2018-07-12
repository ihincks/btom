import numpy as np

__all__ = [
    'TomographySampler', 'StanTomographySampler',
    'BinomialGinibreSampler', 'PoissonGinibreSampler'
]

class TomographySampler(object):
    pass

class StanTomographySampler(TomographySampler):
    pass

class BinomialGinibreSampler(StanTomographySampler):
    pass

class PoissonGinibreSampler(StanTomographySampler):
    pass
