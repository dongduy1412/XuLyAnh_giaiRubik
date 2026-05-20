import argparse
from itertools import product
from pathlib import Path

from rubik.color_profile import build_center_cell_profile, profile_to_json
from rubik.color_recognizer import color_counts, recognize_faces, validate_color_counts, validate_reference_separation
from rubik.face_detector import split_face_into_cells
from rubik.image_processor import load_face_image, preprocess_face
from rubik.rubik_state import FACE_ORDER, build_state_string, rotate_faces
from rubik.solver import parse_solution, solve_cube, verify_cube
from rubik.visualizer import save_unfolded_cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize a 3x3 Rubik cube state from 6 face images and solve it")
    for face in FACE_ORDER:
        parser.add_argument(f"--{face}", required=True, help=f"Image path for face {face}")
    parser.add_argument("--output", default="results/rubik", help="Output directory")
    parser.add_argument("--size", type=int, default=600, help="Normalized face image size")
    parser.add_argument("--equalize", action="store_true", help="Apply CLAHE lighting normalization")
    parser.add_argument("--auto-crop", action="store_true", help="Detect and crop the Rubik face before splitting the grid")
    parser.add_argument("--skip-solve", action="store_true", help="Only recognize and validate colors")
    parser.add_argument("--auto-orient", action="store_true", help="Try 90-degree face rotations until the cube state is solvable")
    for face in FACE_ORDER:
        parser.add_argument(
            f"--rotate-{face}",
            type=int,
            default=0,
            choices=[0, 90, 180, 270],
            help=f"Rotate recognized face {face} clockwise before building the state",
        )
    return parser.parse_args()


def load_cells(args: argparse.Namespace) -> dict[str, list]:
    cells_by_face = {}
    for face in FACE_ORDER:
        path = getattr(args, face)
        image = load_face_image(path)
        face_image = preprocess_face(image, size=args.size, equalize=args.equalize, auto_crop=args.auto_crop)
        cells_by_face[face] = split_face_into_cells(face_image)
    return cells_by_face


def get_manual_rotations(args: argparse.Namespace) -> dict[str, int]:
    return {face: getattr(args, f"rotate_{face}") for face in FACE_ORDER}


def find_solvable_orientation(labels_by_face: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, int], str]:
    for values in product((0, 90, 180, 270), repeat=len(FACE_ORDER)):
        rotations = dict(zip(FACE_ORDER, values, strict=True))
        rotated_labels = rotate_faces(labels_by_face, rotations)
        state = build_state_string(rotated_labels)
        if verify_cube(state) == 0:
            return rotated_labels, rotations, state
    raise ValueError("Cannot find a solvable orientation. Check that all 6 photos are from the same cube state.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_by_face = load_cells(args)
    references = build_center_cell_profile(cells_by_face)
    labels_by_face, details_by_face = recognize_faces(cells_by_face, references=references)
    counts = color_counts(labels_by_face)

    print("Learned center-cell color references:")
    for face, hsv in profile_to_json(references).items():
        print(f"  {face}: H={hsv[0]:.1f}, S={hsv[1]:.1f}, V={hsv[2]:.1f}")
    for warning in validate_reference_separation(references):
        print(f"Warning: {warning}")

    print("Detected color counts:")
    for face in FACE_ORDER:
        print(f"  {face}: {counts.get(face, 0)}")

    validate_color_counts(labels_by_face)
    if args.auto_orient:
        labels_by_face, rotations, state = find_solvable_orientation(labels_by_face)
        print("Auto orientation:")
        for face in FACE_ORDER:
            print(f"  {face}: rotate {rotations[face]} degrees")
    else:
        rotations = get_manual_rotations(args)
        labels_by_face = rotate_faces(labels_by_face, rotations)
        state = build_state_string(labels_by_face)

    print(f"Detected state: {state}")
    print("Validation: OK")

    visualization_path = output_dir / "detected_cube.png"
    save_unfolded_cube(visualization_path, labels_by_face)
    print(f"Saved visualization: {visualization_path}")

    details_path = output_dir / "recognition_details.txt"
    with details_path.open("w", encoding="utf-8") as file:
        for face in FACE_ORDER:
            file.write(f"[{face}]\n")
            for detail in details_by_face[face]:
                file.write(str(detail) + "\n")
    print(f"Saved recognition details: {details_path}")

    if args.skip_solve:
        return

    solution = solve_cube(state)
    moves = parse_solution(solution)
    print(f"Solution: {solution if solution else '(already solved)'}")
    print(f"Moves: {len(moves)}")


if __name__ == "__main__":
    main()
