from collections import Counter


FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
VALID_SYMBOLS = set(FACE_ORDER)
SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def validate_face_structure(labels_by_face: dict[str, list[str]]) -> None:
    missing = [face for face in FACE_ORDER if face not in labels_by_face]
    if missing:
        raise ValueError(f"Missing faces: {missing}")

    for face in FACE_ORDER:
        labels = labels_by_face[face]
        if len(labels) != 9:
            raise ValueError(f"Face {face} must contain exactly 9 labels, got {len(labels)}")
        invalid = [label for label in labels if label not in VALID_SYMBOLS]
        if invalid:
            raise ValueError(f"Face {face} contains invalid labels: {invalid}")


def rotate_face_labels(labels: list[str], degrees: int) -> list[str]:
    normalized = degrees % 360
    if normalized not in {0, 90, 180, 270}:
        raise ValueError("Face rotation must be one of 0, 90, 180, 270 degrees")

    rotated = list(labels)
    for _ in range(normalized // 90):
        rotated = [rotated[index] for index in (6, 3, 0, 7, 4, 1, 8, 5, 2)]
    return rotated


def rotate_faces(labels_by_face: dict[str, list[str]], rotations: dict[str, int]) -> dict[str, list[str]]:
    validate_face_structure(labels_by_face)
    return {
        face: rotate_face_labels(labels_by_face[face], rotations.get(face, 0))
        for face in FACE_ORDER
    }


def build_state_string(labels_by_face: dict[str, list[str]]) -> str:
    validate_face_structure(labels_by_face)
    state = "".join("".join(labels_by_face[face]) for face in FACE_ORDER)
    validate_state_string(state)
    return state


def validate_state_string(state: str) -> None:
    if len(state) != 54:
        raise ValueError(f"State string must contain 54 characters, got {len(state)}")

    invalid_symbols = sorted(set(state) - VALID_SYMBOLS)
    if invalid_symbols:
        raise ValueError(f"State contains invalid symbols: {invalid_symbols}")

    counts = Counter(state)
    invalid_counts = {face: counts.get(face, 0) for face in FACE_ORDER if counts.get(face, 0) != 9}
    if invalid_counts:
        raise ValueError(f"Invalid state counts: {invalid_counts}. Each symbol must appear exactly 9 times.")
