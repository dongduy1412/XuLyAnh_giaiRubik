import importlib
import sys
from pathlib import Path

from .rubik_state import validate_state_string


def _load_kociemba_module():
    try:
        module = importlib.import_module("kociemba")
        if hasattr(module, "solve"):
            return module
    except ModuleNotFoundError:
        pass

    local_package_root = Path(__file__).resolve().parents[2] / "kociemba"
    if local_package_root.exists():
        sys.path.insert(0, str(local_package_root))
        sys.modules.pop("kociemba", None)
        module = importlib.import_module("kociemba")
        if hasattr(module, "solve"):
            return module

    raise ImportError("Cannot find kociemba.solve. Install kociemba or keep the local kociemba package folder.")


def verify_cube(state_string: str) -> int:
    validate_state_string(state_string)
    _load_kociemba_module()
    tools = importlib.import_module("kociemba.pykociemba.tools")
    return tools.verify(state_string)


def solve_cube(state_string: str, max_depth: int = 24) -> str:
    validate_state_string(state_string)
    kociemba = _load_kociemba_module()
    try:
        return kociemba.solve(state_string, max_depth=max_depth).strip()
    except ValueError as error:
        raise ValueError(f"Detected cube state is invalid or unsolvable: {error}") from error


def parse_solution(solution: str) -> list[str]:
    return [move for move in solution.split() if move]
