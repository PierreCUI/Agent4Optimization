from .check_tools import check_tools
from .install_python import setup_environment
from .run_autofreefem import run_autofreefem
from .run_freefem_linux import run_freefem as run_freefem_linux
from .run_freefem_windows import run_freefem as run_freefem_windows
from .runner import run_problem

__all__ = [
    "run_problem",
    "check_tools",
    "setup_environment",
    "run_autofreefem",
    "run_freefem_linux",
    "run_freefem_windows",
]
