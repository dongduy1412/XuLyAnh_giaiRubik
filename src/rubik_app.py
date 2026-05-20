from copy import deepcopy
from io import BytesIO
from itertools import product
from pathlib import Path

import cv2
from PIL import Image, ImageOps
import streamlit as st

from rubik.color_profile import load_reference_profile, profile_to_json
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


def get_palette_to_face(exclude_face: str | None = None) -> dict[str, str]:
    mapping = {}
    for face, result in st.session_state["face_results"].items():
        if face == exclude_face:
            continue
        center_palette = result.get("center_palette")
        if isinstance(center_palette, str):
            mapping[center_palette] = face
    return mapping


def get_runtime_references(exclude_face: str | None = None) -> tuple[dict[str, object], dict[str, str], int]:
    palette = load_reference_profile()
    palette_to_face = get_palette_to_face(exclude_face)
    return palette, palette_to_face, len(palette_to_face)




def recognize_single_face(
    face: str,
    image_path: Path,
    size: int,
    equalize: bool,
    auto_crop: bool,
    palette: dict[str, object],
    palette_to_face: dict[str, str],
) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_face_image(image_path)
    face_image = preprocess_face(image, size=size, equalize=equalize, auto_crop=auto_crop)
    cells = split_face_into_cells(face_image)

    center_lab = get_mean_lab(cells[4])
    center_palette, _, _, _ = classify_cell(center_lab, palette)
    updated_palette_to_face = dict(palette_to_face)
    updated_palette_to_face[center_palette] = face

    labels = []
    palette_labels = []
    details = []
    for index, cell in enumerate(cells):
        lab = get_mean_lab(cell)
        palette_label, distance, confidence_gap, distances = classify_cell(lab, palette)
        label = updated_palette_to_face.get(palette_label, palette_label)
        labels.append(label)
        palette_labels.append(palette_label)
        details.append({
            "index": index,
            "label": label,
            "palette": palette_label,
            "distance": round(distance, 3),
            "confidence_gap": round(confidence_gap, 3),
            "distances": {key: round(value, 3) for key, value in distances.items()},
        })

    labels[4] = face
    palette_labels[4] = center_palette
    details[4]["label"] = face
    details[4]["palette"] = center_palette

    debug_dir = OUTPUT_DIR / "debug_faces"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{face}.png"
    cv2.imwrite(str(debug_path), draw_grid(face_image))

    return {
        "face": face,
        "image_path": str(image_path),
        "debug_path": str(debug_path),
        "labels": labels,
        "palette_labels": palette_labels,
        "center_palette": center_palette,
        "details": details,
        "references": profile_to_json(palette),
        "cells": cells,
    }


def color_map_from_palette_mapping(palette: dict[str, object], palette_to_face: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    references = dict(palette)
    references.update({
        face: palette[color_key]
        for color_key, face in palette_to_face.items()
        if face in FACE_ORDER and color_key in palette
    })
    return build_color_map_from_references(references)


def get_current_color_map() -> dict[str, tuple[int, int, int]]:
    try:
        return color_map_from_palette_mapping(load_reference_profile(), get_palette_to_face())
    except Exception:
        return build_color_map_from_references({})


def color_map_from_result(result: dict[str, object]) -> dict[str, tuple[int, int, int]]:
    import numpy as np
    palette = {
        color_key: np.array(values, dtype=np.float32)
        for color_key, values in result.get("references", {}).items()
    }
    palette_to_face = get_palette_to_face(exclude_face=result.get("face"))
    center_palette = result.get("center_palette")
    face = result.get("face")
    if isinstance(center_palette, str) and face in FACE_ORDER:
        palette_to_face[center_palette] = face
    return color_map_from_palette_mapping(palette, palette_to_face)


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
        render_face_image(result["labels"], get_current_color_map())
        st.caption(f"Palette tâm: {result.get('center_palette', '-')}")
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


def render_palette_legend(palette_keys: list[str], color_map: dict) -> None:
    """Show a color legend so users know what each palette key looks like."""
    cols = st.columns(len(palette_keys))
    for col, key in zip(cols, palette_keys):
        bgr = color_map.get(key, (128, 128, 128))
        r, g, b = bgr[2], bgr[1], bgr[0]
        with col:
            st.markdown(
                f'<div style="background:rgb({r},{g},{b});color:{"#000" if (r*0.299+g*0.587+b*0.114)>140 else "#fff"};'
                f'text-align:center;padding:6px;border-radius:6px;font-weight:bold;">{key}</div>',
                unsafe_allow_html=True,
            )


def render_face_correction(face: str, palette_labels: list[str], palette_keys: list[str]) -> None:
    """Render correction selectboxes using palette keys as options."""
    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            index = row * 3 + col
            current = palette_labels[index]
            selected_index = palette_keys.index(current) if current in palette_keys else 0
            with cols[col]:
                st.selectbox(
                    f"{face}{index + 1}",
                    options=palette_keys,
                    index=selected_index,
                    key=f"pending_{face}_{index}",
                    label_visibility="collapsed",
                )


def read_pending_labels(face: str) -> list[str]:
    return [st.session_state[f"pending_{face}_{index}"] for index in range(9)]


def render_pending_result(face: str, result: dict[str, object]) -> None:
    st.divider()
    st.subheader("Kết quả nhận diện")
    cmap = color_map_from_result(result)
    left, right = st.columns(2)
    with left:
        st.caption("Ảnh đã crop và kẻ lưới")
        st.image(result["debug_path"], use_container_width=True)
    with right:
        st.caption("Màu nhận diện")
        palette_labels = result.get("palette_labels", result["labels"])
        render_face_image(palette_labels, cmap)

    palette_keys = sorted(result.get("references", {}).keys())
    center_palette = result.get("center_palette", palette_labels[4] if len(palette_labels) > 4 else "-")
    st.info(f"Ô giữa = {center_palette} → gán mặt {face}. Nếu màu sai, sửa trong lưới bên dưới.")

    st.subheader("Sửa màu nếu nhận diện sai")
    render_palette_legend(palette_keys, cmap)
    with st.form(f"confirm_{face}"):
        render_face_correction(face, palette_labels, palette_keys)
        submitted = st.form_submit_button("Thêm vào layout", use_container_width=True)

    if submitted:
        corrected = deepcopy(result)
        corrected_palette = read_pending_labels(face)
        corrected["labels"] = corrected_palette
        corrected["palette_labels"] = corrected_palette
        corrected["center_palette"] = corrected_palette[4] if len(corrected_palette) > 4 else center_palette
        st.session_state["face_results"][face] = corrected
        go_home()

    with st.expander("Chi tiết confidence"):
        rows = []
        for detail in result["details"]:
            rows.append({
                "Ô": detail["index"] + 1,
                "Palette": detail.get("palette", detail["label"]),
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
        auto_crop = st.checkbox("Tự cắt viền mặt Rubik", value=True)

    uploaded_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"], key=f"upload_{face}")
    camera_file = st.camera_input("Hoặc chụp ảnh", key=f"camera_{face}")
    selected_file = camera_file or uploaded_file

    if selected_file is not None:
        st.caption(f"Tên ảnh: {selected_file.name}")
        st.image(normalize_uploaded_image(selected_file), caption="Ảnh đã chọn", use_container_width=True)
        if st.button("Nhận diện màu", type="primary", use_container_width=True):
            try:
                image_path = save_uploaded_face(face, selected_file)
                palette, palette_to_face, captured_count = get_runtime_references(exclude_face=face)
                st.session_state["pending_result"] = recognize_single_face(face, image_path, size, equalize, auto_crop, palette, palette_to_face)
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
    """Convert stored palette labels to face labels using center cell mapping.

    This only works correctly when all 6 faces have been uploaded,
    because it needs all 6 center palette->face mappings.
    """
    palette_to_face = get_palette_to_face()
    result = {}
    for face in FACE_ORDER:
        stored = st.session_state["face_results"][face]["labels"]
        face_labels = [palette_to_face.get(label, label) for label in stored]
        face_labels[4] = face  # enforce center
        result[face] = face_labels
    return result


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
            save_unfolded_cube(output_path, labels_by_face, color_map=get_current_color_map())

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
