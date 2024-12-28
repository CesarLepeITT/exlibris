#def initialize():
#    return pickle.loads(importlib_resources.read_binary(__name__, 'chartsdata.pkl'))

#def load_datasets():
#    return importlib_resources.read_text(__name__, 'stats.csv')

from .stats import (
    Stats
)
from .dataset import (
    Dataset
)

from .stats_gsgp import (
    stats_gsgp
)