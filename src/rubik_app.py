from copy import deepcopy
from io import BytesIO
from itertools import product
from pathlib import Path

import cv2
from PIL import Image, ImageOps
import streamlit as st

from rubik.color_profile import build_center_cell_profile, load_reference_profile, profile_to_json
from rubik.color_recognizer import classify_cell, color_counts, get_mean_lab, validate_color_counts
from rubik.face_detector import draw_grid, split_face_into_cells
from rubik.image_processor import load_face_image, preprocess_face
from rubik.rubik_state import FACE_ORDER, build_state_string, rotate_faces
from rubik.solver import parse_solution, solve_cube, verify_cube
from rubik.visualizer import build_color_map_from_references, draw_face, save_unfolded_cube

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/rubik")
FACE_LABELS = {
    "U": "Mặt trên",
    "R": "Mặt phải",
    "F": "Mặt trước",
    "D": "Mặt dưới",
    "L": "Mặt trái",
    "B": "Mặt sau",
}


def init_state() -> None:
    defaults = {
        "page": "home",
        "active_face": None,
        "face_results": {},
        "pending_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_uploaded_image(uploaded_file: object) -> Image.Image:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    return ImageOps.exif_transpose(image).convert("RGB")


def save_uploaded_face(face: str, uploaded_file: object) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{face}.jpg"
    image = normalize_uploaded_image(uploaded_file)
    image.save(path, format="JPEG", quality=95)
    return path


def get_runtime_references(exclude_face: str | None = None) -> tuple[dict[str, object], int]:
    references = load_reference_profile()
    captured_cells = {
        face: result["cells"]
        for face, result in st.session_state["face_results"].items()
        if face != exclude_face and "cells" in result
    }
    if captured_cells:
        captured_references = build_center_cell_profile(captured_cells)
        references.update(captured_references)
    return references, len(captured_cells)


def recognize_single_face(face: str, image_path: Path, size: int, equalize: bool, auto_crop: bool, references: dict[str, object]) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_face_image(image_path)
    face_image = preprocess_face(image, size=size, equalize=equalize, auto_crop=auto_crop)
    cells = split_face_into_cells(face_image)

    # Key insight: center cell (index 4) of this face IS the reference color
    # for this face label. Override the reference before classifying other cells.
    center_lab = get_mean_lab(cells[4])
    updated_references = dict(references)
    updated_references[face] = center_lab

    labels = []
    details = []
    for index, cell in enumerate(cells):
        lab = get_mean_lab(cell)
        label, distance, confidence_gap, distances = classify_cell(lab, updated_references)
        labels.append(label)
        details.append({
            "index": index,
            "label": label,
            "distance": round(distance, 3),
            "confidence_gap": round(confidence_gap, 3),
            "distances": {key: round(value, 3) for key, value in distances.items()},
        })

    # Force center cell to always be labeled as its face (center is fixed on Rubik 3x3)
    labels[4] = face
    details[4]["label"] = face

    debug_dir = OUTPUT_DIR / "debug_faces"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{face}.png"
    cv2.imwrite(str(debug_path), draw_grid(face_image))

    return {
        "face": face,
        "image_path": str(image_path),
        "debug_path": str(debug_path),
        "labels": labels,
        "details": details,
        "references": profile_to_json(updated_references),
        "cells": cells,
    }


def get_current_color_map() -> dict[str, tuple[int, int, int]]:
    """Build a color map from all currently stored face results."""
    import numpy as np
    all_refs = {}
    for face, result in st.session_state.get("face_results", {}).items():
        refs = result.get("references", {})
        if face in refs:
            all_refs[face] = np.array(refs[face], dtype=np.float32)
    # Also merge static reference for faces not yet captured
    try:
        static_refs = load_reference_profile()
        for face in FACE_ORDER:
            if face not in all_refs and face in static_refs:
                all_refs[face] = static_refs[face]
    except Exception:
        pass
    return build_color_map_from_references(all_refs)


def render_face_image(labels: list[str], color_map: dict | None = None) -> None:
    if color_map is None:
        color_map = get_current_color_map()
    image = draw_face(labels, cell_size=55, color_map=color_map)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)


def go_home() -> None:
    st.session_state["page"] = "home"
    st.session_state["active_face"] = None
    st.session_state["pending_result"] = None
    st.rerun()


def open_face_page(face: str) -> None:
    st.session_state["page"] = "face"
    st.session_state["active_face"] = face
    existing = st.session_state["face_results"].get(face)
    st.session_state["pending_result"] = deepcopy(existing) if existing else None
    st.rerun()


def render_slot(face: str) -> None:
    result = st.session_state["face_results"].get(face)
    st.markdown(f"**{face} - {FACE_LABELS[face]}**")
    if result:
        render_face_image(result["labels"])
        if st.button("Sửa mặt", key=f"edit_{face}", use_container_width=True):
            open_face_page(face)
    else:
        st.markdown(
            """
            <div style="height:145px;border:2px dashed #999;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:42px;color:#777;">+</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Thêm {face}", key=f"add_{face}", use_container_width=True):
            open_face_page(face)


def render_cube_layout() -> None:
    st.subheader("Layout Rubik")
    st.caption("Bấm vào dấu cộng để thêm từng mặt. Mặt đã nhận diện sẽ hiện dạng lưới 3x3.")

    row_u = st.columns([1, 1, 1, 1], gap="medium")
    with row_u[1]:
        render_slot("U")

    row_middle = st.columns(4, gap="medium")
    for column, face in zip(row_middle, ["L", "F", "R", "B"], strict=True):
        with column:
            render_slot(face)

    row_d = st.columns([1, 1, 1, 1], gap="medium")
    with row_d[1]:
        render_slot("D")


def render_face_correction(face: str, labels: list[str]) -> None:
    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            index = row * 3 + col
            with cols[col]:
                st.selectbox(
                    f"{face}{index + 1}",
                    options=FACE_ORDER,
                    index=FACE_ORDER.index(labels[index]),
                    key=f"pending_{face}_{index}",
                    label_visibility="collapsed",
                )


def read_pending_labels(face: str) -> list[str]:
    return [st.session_state[f"pending_{face}_{index}"] for index in range(9)]


def render_pending_result(face: str, result: dict[str, object]) -> None:
    st.divider()
    st.subheader("Kết quả nhận diện")
    left, right = st.columns(2)
    with left:
        st.caption("Ảnh đã crop và kẻ lưới")
        st.image(result["debug_path"], use_container_width=True)
    with right:
        st.caption("Màu nhận diện")
        render_face_image(result["labels"])

    st.subheader("Sửa màu nếu nhận diện sai")
    with st.form(f"confirm_{face}"):
        render_face_correction(face, result["labels"])
        submitted = st.form_submit_button("Thêm vào layout", use_container_width=True)

    if submitted:
        corrected = deepcopy(result)
        corrected["labels"] = read_pending_labels(face)
        st.session_state["face_results"][face] = corrected
        go_home()

    with st.expander("Chi tiết confidence"):
        rows = []
        for detail in result["details"]:
            rows.append({
                "Ô": detail["index"] + 1,
                "Nhãn": detail["label"],
                "Khoảng cách": detail["distance"],
                "Confidence gap": detail["confidence_gap"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_face_page() -> None:
    face = st.session_state["active_face"]
    if face is None:
        go_home()

    top_left, top_right = st.columns([1, 3])
    with top_left:
        if st.button("Quay lại", use_container_width=True):
            go_home()
    with top_right:
        st.subheader(f"Thêm ảnh cho mặt {face} - {FACE_LABELS[face]}")

    with st.sidebar:
        st.header("Tùy chọn nhận diện")
        size = st.slider("Kích thước chuẩn hóa", min_value=300, max_value=900, value=600, step=100)
        equalize = st.checkbox("Cân bằng sáng CLAHE", value=False)
        auto_crop = st.checkbox("Tự crop mặt Rubik", value=True)

    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"], key=f"upload_{face}")
    camera_file = st.camera_input("Hoặc chụp ảnh", key=f"camera_{face}")
    selected_file = camera_file or uploaded_file

    if selected_file is not None:
        st.caption(f"Tên ảnh: {selected_file.name}")
        st.image(normalize_uploaded_image(selected_file), caption="Ảnh đã chọn", use_container_width=True)
        if st.button("Nhận diện màu", type="primary", use_container_width=True):
            try:
                image_path = save_uploaded_face(face, selected_file)
                references, captured_count = get_runtime_references(exclude_face=face)
                st.session_state["pending_result"] = recognize_single_face(face, image_path, size, equalize, auto_crop, references)
                st.session_state["pending_reference_count"] = captured_count
                st.rerun()
            except Exception as error:
                st.error(str(error))
                st.warning("Hãy thử ảnh rõ hơn, chụp thẳng mặt Rubik và đủ sáng.")

    pending_result = st.session_state.get("pending_result")
    if pending_result:
        st.info(f"Đang hiệu chỉnh bằng {st.session_state.get('pending_reference_count', 0)} mặt đã lưu trong layout.")
        render_pending_result(face, pending_result)


def labels_from_layout() -> dict[str, list[str]]:
    return {
        face: st.session_state["face_results"][face]["labels"]
        for face in FACE_ORDER
    }


def find_solvable_orientation(labels_by_face: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, int], str]:
    for values in product((0, 90, 180, 270), repeat=len(FACE_ORDER)):
        rotations = dict(zip(FACE_ORDER, values, strict=True))
        rotated_labels = rotate_faces(labels_by_face, rotations)
        state = build_state_string(rotated_labels)
        if verify_cube(state) == 0:
            return rotated_labels, rotations, state
    raise ValueError("Không tìm được hướng xoay hợp lệ. Hãy kiểm tra nhãn màu và hướng của 6 mặt.")


def render_solver_panel() -> None:
    completed = [face for face in FACE_ORDER if face in st.session_state["face_results"]]
    missing = [face for face in FACE_ORDER if face not in st.session_state["face_results"]]

    st.subheader("Trạng thái")
    st.write(f"Đã thêm: {len(completed)}/6 mặt")
    if missing:
        st.info(f"Còn thiếu: {', '.join(missing)}")
        return

    labels_by_face = labels_from_layout()
    counts = color_counts(labels_by_face)
    st.write("Số lượng màu:", {face: counts.get(face, 0) for face in FACE_ORDER})

    auto_orient = st.checkbox("Tự xoay hướng mặt để tìm trạng thái hợp lệ", value=True)
    if st.button("Kiểm tra và giải Rubik", type="primary", use_container_width=True):
        try:
            validate_color_counts(labels_by_face)
            rotations = {face: 0 for face in FACE_ORDER}
            if auto_orient:
                labels_by_face, rotations, state = find_solvable_orientation(labels_by_face)
            else:
                state = build_state_string(labels_by_face)
                verify_code = verify_cube(state)
                if verify_code != 0:
                    raise ValueError(f"Kociemba verify failed with code {verify_code}")

            solution = solve_cube(state)
            moves = parse_solution(solution)
            output_path = OUTPUT_DIR / "detected_cube_final.png"
            save_unfolded_cube(output_path, labels_by_face)

            st.success("Trạng thái Rubik hợp lệ")
            st.image(str(output_path), caption="Layout cuối cùng", use_container_width=True)
            st.write("Hướng xoay:", {face: f"{degrees}°" for face, degrees in rotations.items()})
            st.code(state, language="text")
            st.subheader("Lời giải")
            st.code(solution if solution else "Rubik đã giải sẵn", language="text")
            st.write(f"Số bước: {len(moves)}")
        except Exception as error:
            st.error(str(error))
            st.warning("Hãy kiểm tra lại màu từng ô hoặc chụp lại mặt bị sai.")


def render_home() -> None:
    st.title("Rubik Cube Solver bằng xử lý ảnh")
    st.write("Thêm từng mặt Rubik vào layout, kiểm tra màu nhận diện, rồi giải khi đủ 6 mặt.")

    with st.expander("Quy tắc chụp ảnh", expanded=True):
        st.markdown(
            """
- Chọn một mặt làm **F - mặt trước / mặt đối diện camera**.
- Xác định các mặt còn lại theo F: U ở trên, D ở dưới, L bên trái, R bên phải, B phía sau.
- Mỗi ảnh nên chụp thẳng một mặt, Rubik chiếm phần lớn khung hình, ánh sáng đều.
- Nếu màu nhận diện sai, sửa trực tiếp trong lưới 3x3 trước khi thêm vào layout.
"""
        )

    render_cube_layout()
    render_solver_panel()

    if st.button("Xóa layout hiện tại"):
        st.session_state["face_results"] = {}
        st.session_state["pending_result"] = None
        st.rerun()


st.set_page_config(page_title="Rubik Image Solver", page_icon="R", layout="wide")
init_state()

if st.session_state["page"] == "face":
    render_face_page()
else:
    render_home()
