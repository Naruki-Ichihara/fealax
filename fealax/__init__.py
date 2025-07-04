from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

# Import modules
from . import mesh
from . import problem
from . import solver
from . import basis
from . import utils
from . import memory_utils
from . import large_solver

__version__ = "0.0.1"