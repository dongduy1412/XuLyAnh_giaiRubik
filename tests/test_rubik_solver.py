from rubik.rubik_state import SOLVED_STATE
from rubik.solver import parse_solution, solve_cube


def test_parse_solution() -> None:
    assert parse_solution("R U R' U'") == ["R", "U", "R'", "U'"]


def test_solve_solved_cube_returns_string() -> None:
    solution = solve_cube(SOLVED_STATE)
    assert isinstance(solution, str)
