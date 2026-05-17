from pathlib import Path

import streamlit as st

from rubik.color_recognizer import color_counts, recognize_faces, validate_color_counts
from rubik.face_detector import split_face_into_cells
from rubik.image_processor import load_face_image, preprocess_face
from rubik.rubik_state import FACE_ORDER, build_state_string, rotate_faces
from rubik.solver import parse_solution, solve_cube, verify_cube
from rubik.visualizer import save_unfolded_cube

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/rubik")
FACE_LABELS = {
    "F": "F - Mặt trước / đối diện camera",
    "U": "U - Mặt trên của F",
    "D": "D - Mặt dưới của F",
    "L": "L - Mặt trái của F",
    "R": "R - Mặt phải của F",
    "B": "B - Mặt sau lưng F",
}
CAPTURE_ORDER = ["F", "U", "D", "L", "R", "B"]


def save_uploaded_faces(uploaded_faces: dict[str, object]) -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for face, uploaded_file in uploaded_faces.items():
        path = DATA_DIR / f"{face}.jpg"
        path.write_bytes(uploaded_file.getvalue())
        paths[face] = path
    return paths


def load_cells(image_paths: dict[str, Path], size: int = 600, equalize: bool = False) -> dict[str, list]:
    cells_by_face = {}
    for face in FACE_ORDER:
        image = load_face_image(image_paths[face])
        face_image = preprocess_face(image, size=size, equalize=equalize)
        cells_by_face[face] = split_face_into_cells(face_image)
    return cells_by_face


def find_solvable_orientation(labels_by_face: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, int], str]:
    from itertools import product

    for values in product((0, 90, 180, 270), repeat=len(FACE_ORDER)):
        rotations = dict(zip(FACE_ORDER, values, strict=True))
        rotated_labels = rotate_faces(labels_by_face, rotations)
        state = build_state_string(rotated_labels)
        if verify_cube(state) == 0:
            return rotated_labels, rotations, state
    raise ValueError("Không tìm được hướng xoay hợp lệ. Hãy kiểm tra 6 ảnh có đúng cùng một trạng thái Rubik và đúng mặt F/U/D/L/R/B không.")


def solve_from_images(image_paths: dict[str, Path], size: int, equalize: bool, auto_orient: bool) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells_by_face = load_cells(image_paths, size=size, equalize=equalize)
    labels_by_face, details_by_face = recognize_faces(cells_by_face)
    counts = color_counts(labels_by_face)
    validate_color_counts(labels_by_face)

    rotations = {face: 0 for face in FACE_ORDER}
    if auto_orient:
        labels_by_face, rotations, state = find_solvable_orientation(labels_by_face)
    else:
        state = build_state_string(labels_by_face)

    visualization_path = OUTPUT_DIR / "detected_cube.png"
    save_unfolded_cube(visualization_path, labels_by_face)

    details_path = OUTPUT_DIR / "recognition_details.txt"
    with details_path.open("w", encoding="utf-8") as file:
        for face in FACE_ORDER:
            file.write(f"[{face}]\n")
            for detail in details_by_face[face]:
                file.write(str(detail) + "\n")

    solution = solve_cube(state)
    moves = parse_solution(solution)
    return {
        "counts": counts,
        "rotations": rotations,
        "state": state,
        "solution": solution,
        "moves": moves,
        "visualization_path": visualization_path,
        "details_path": details_path,
    }


st.set_page_config(page_title="Rubik Image Solver", page_icon="🧩", layout="wide")
st.title("Rubik Cube Solver bằng xử lý ảnh")
st.write("Dùng điện thoại mở trang này, chụp hoặc upload lần lượt 6 mặt Rubik. App sẽ tự lưu đúng tên `F.jpg`, `U.jpg`, `D.jpg`, `L.jpg`, `R.jpg`, `B.jpg`.")

with st.expander("Quy tắc chụp ảnh", expanded=True):
    st.markdown(
        """
- Chọn một mặt làm **F - mặt trước / mặt đối diện camera**.
- Giữ quy ước đó để xác định **U, D, L, R, B**.
- Layout solver chuẩn: `U` ở trên, hàng giữa là `L F R B`, `D` ở dưới.
- Ảnh nên chụp thẳng, Rubik chiếm phần lớn ảnh, ánh sáng đều.
"""
    )

uploaded_faces = {}
cols = st.columns(2)
for index, face in enumerate(CAPTURE_ORDER):
    with cols[index % 2]:
        uploaded_file = st.file_uploader(
            FACE_LABELS[face],
            type=["jpg", "jpeg", "png"],
            key=f"upload_{face}",
        )
        if uploaded_file is not None:
            uploaded_faces[face] = uploaded_file
            st.image(uploaded_file, caption=f"Đã chọn {face}", use_container_width=True)

with st.sidebar:
    st.header("Tùy chọn")
    size = st.slider("Kích thước chuẩn hóa mỗi mặt", min_value=300, max_value=900, value=600, step=100)
    equalize = st.checkbox("Cân bằng sáng CLAHE", value=False)
    auto_orient = st.checkbox("Tự xoay hướng mặt cho solver", value=True)

ready = all(face in uploaded_faces for face in CAPTURE_ORDER)
if not ready:
    missing = [face for face in CAPTURE_ORDER if face not in uploaded_faces]
    st.info(f"Cần thêm ảnh: {', '.join(missing)}")

if st.button("Nhận diện và giải Rubik", type="primary", disabled=not ready):
    try:
        image_paths = save_uploaded_faces(uploaded_faces)
        result = solve_from_images(image_paths, size=size, equalize=equalize, auto_orient=auto_orient)

        st.success("Nhận diện và giải thành công")
        st.subheader("Số lượng màu nhận diện")
        st.write({face: result["counts"].get(face, 0) for face in FACE_ORDER})

        st.subheader("Hướng xoay tự động")
        st.write({face: f"{degrees}°" for face, degrees in result["rotations"].items()})

        st.subheader("Layout nhận diện")
        st.image(str(result["visualization_path"]), use_container_width=True)

        st.subheader("State Kociemba")
        st.code(result["state"])

        st.subheader("Lời giải")
        st.code(result["solution"] if result["solution"] else "Rubik đã giải sẵn")
        st.write(f"Số bước: {len(result['moves'])}")
        st.caption(f"Chi tiết nhận diện: {result['details_path']}")
    except Exception as error:
        st.error(str(error))
        st.warning("Nếu lỗi cube invalid, hãy kiểm tra lại tên mặt F/U/D/L/R/B hoặc chụp lại đủ 6 mặt trong cùng một trạng thái Rubik.")
