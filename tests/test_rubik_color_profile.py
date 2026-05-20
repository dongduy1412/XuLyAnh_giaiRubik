import numpy as np

from rubik.color_profile import _assign_samples_to_faces, build_center_cell_profile, build_reference_profile
from rubik.rubik_state import FACE_ORDER


def test_build_center_cell_profile_uses_face_centers(monkeypatch) -> None:
    cells_by_face = {}
    expected = {}
    for face_index, face in enumerate(FACE_ORDER):
        cells = []
        for cell_index in range(9):
            marker = face_index * 10 + cell_index
            cells.append(np.full((4, 4, 3), marker, dtype=np.uint8))
        cells_by_face[face] = cells
        expected[face] = np.array([face_index * 10 + 4, 100.0, 200.0], dtype=np.float32)

    def fake_get_mean_hsv(cell: np.ndarray) -> np.ndarray:
        marker = float(cell[0, 0, 0])
        return np.array([marker, 100.0, 200.0], dtype=np.float32)

    monkeypatch.setattr("rubik.color_profile.get_mean_lab", fake_get_mean_hsv)

    profile = build_center_cell_profile(cells_by_face)

    for face in FACE_ORDER:
        assert np.allclose(profile[face], expected[face])


def test_assign_samples_to_faces_matches_standard_cube_colors() -> None:
    samples = [
        np.array([15.0, 5.0, 245.0], dtype=np.float32),
        np.array([2.0, 240.0, 240.0], dtype=np.float32),
        np.array([62.0, 225.0, 220.0], dtype=np.float32),
        np.array([29.0, 210.0, 242.0], dtype=np.float32),
        np.array([17.0, 235.0, 228.0], dtype=np.float32),
        np.array([108.0, 215.0, 222.0], dtype=np.float32),
    ]
    profile = _assign_samples_to_faces(list(reversed(samples)))

    assert np.allclose(profile["U"], samples[0])
    assert np.allclose(profile["R"], samples[1])
    assert np.allclose(profile["F"], samples[2])
    assert np.allclose(profile["D"], samples[3])
    assert np.allclose(profile["L"], samples[4])
    assert np.allclose(profile["B"], samples[5])


def test_build_reference_profile_returns_palette_without_face_binding(monkeypatch, tmp_path) -> None:
    image_paths = [tmp_path / f"color_{index}.jpg" for index in range(len(FACE_ORDER))]
    samples = {
        image_paths[0]: np.array([17.0, 235.0, 228.0], dtype=np.float32),
        image_paths[1]: np.array([2.0, 240.0, 240.0], dtype=np.float32),
        image_paths[2]: np.array([62.0, 225.0, 220.0], dtype=np.float32),
        image_paths[3]: np.array([26.0, 5.0, 247.0], dtype=np.float32),
        image_paths[4]: np.array([108.0, 215.0, 222.0], dtype=np.float32),
        image_paths[5]: np.array([29.0, 220.0, 245.0], dtype=np.float32),
    }

    monkeypatch.setattr("rubik.color_profile._collect_calibration_images", lambda calibration_dir: image_paths)
    monkeypatch.setattr("rubik.color_profile._sample_lab", lambda path: samples[path])

    profile = build_reference_profile(tmp_path)

    assert list(profile) == ["C1", "C2", "C3", "C4", "C5", "C6"]
    for index, color_key in enumerate(profile):
        assert profile[color_key]["source"] == str(image_paths[index])
        assert profile[color_key]["lab"] == [float(value) for value in samples[image_paths[index]]]
