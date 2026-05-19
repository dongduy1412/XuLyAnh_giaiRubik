# Rubik Cube Solver bằng Xử lý Ảnh

Nhận diện trạng thái Rubik 3×3 từ 6 ảnh chụp các mặt, sau đó dùng thuật toán **Kociemba (Two-Phase Algorithm)** để đề xuất chuỗi bước giải.

## Tính năng

- Nhận diện màu 54 ô từ 6 ảnh mặt Rubik bằng **OpenCV** (HSV color matching).
- Tự động xoay hướng mặt (`--auto-orient`) để tìm trạng thái hợp lệ.
- Giải Rubik bằng thuật toán **Kociemba** (Two-Phase).
- Xuất sơ đồ Rubik 2D và chi tiết nhận diện.
- **Web app (Streamlit)** — upload/chụp ảnh từ điện thoại, xem kết quả trực tiếp.
- Hỗ trợ cân bằng sáng CLAHE cho ảnh chụp trong điều kiện ánh sáng không đều.

## Pipeline

```text
6 ảnh mặt Rubik
 → crop vuông trung tâm + resize
 → chia đều lưới 3×3
 → lấy màu HSV trung bình vùng giữa từng ô
 → lấy 6 ô trung tâm làm màu tham chiếu
 → gán 54 ô theo khoảng cách màu HSV có trọng số
 → validate mỗi màu xuất hiện đúng 9 lần
 → build chuỗi trạng thái theo thứ tự U R F D L B
 → gọi Kociemba solver
 → xuất sơ đồ Rubik 2D và lời giải
```

## Cấu trúc dự án

```text
XuLyAnh/
├── src/
│   ├── rubik_main.py            # CLI entry point
│   ├── rubik_app.py             # Streamlit web app
│   └── rubik/
│       ├── image_processor.py   # Load & tiền xử lý ảnh
│       ├── face_detector.py     # Chia mặt Rubik thành 9 ô
│       ├── color_recognizer.py  # Nhận diện màu HSV
│       ├── rubik_state.py       # Xây dựng chuỗi trạng thái
│       ├── solver.py            # Gọi Kociemba solver
│       └── visualizer.py        # Xuất sơ đồ 2D
├── data/
│   ├── rubik_samples/           # Ảnh mẫu Rubik
│   └── *.jpg                    # Ảnh mặt U/R/F/D/L/B
├── tests/                       # Unit tests (pytest)
├── docs/
│   └── REPORT.md                # Báo cáo đề tài
└── requirements.txt
```

## Cài đặt

```bash
# Clone repo
git clone https://github.com/dongduy1412/XuLyAnh_giaiRubik.git
cd XuLyAnh_giaiRubik/XuLyAnh

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Yêu cầu

- **Python** ≥ 3.10
- Các thư viện chính: `opencv-python`, `numpy`, `matplotlib`, `streamlit`, `Pillow`
- Xem đầy đủ tại [`requirements.txt`](XuLyAnh/requirements.txt)

## Chuẩn bị ảnh

Đặt 6 ảnh mặt Rubik vào thư mục `data/` hoặc `data/rubik_samples/`:

| File     | Mặt         |
|----------|-------------|
| `U.jpg`  | Mặt trên    |
| `R.jpg`  | Mặt phải    |
| `F.jpg`  | Mặt trước   |
| `D.jpg`  | Mặt dưới    |
| `L.jpg`  | Mặt trái    |
| `B.jpg`  | Mặt sau     |

**Lưu ý khi chụp ảnh:**

- Chụp thẳng vuông góc từng mặt.
- Mặt Rubik chiếm phần lớn ảnh.
- Ánh sáng trắng đều, tránh bóng và lóa.
- Giữ cùng quy ước hướng khi chụp tất cả 6 mặt.

## Sử dụng

### CLI

```bash
cd XuLyAnh

python src/rubik_main.py \
  --U data/U.jpg --R data/R.jpg --F data/F.jpg \
  --D data/D.jpg --L data/L.jpg --B data/B.jpg \
  --output results/rubik --auto-orient
```

**Tùy chọn:**

| Flag             | Mô tả                                             |
|------------------|----------------------------------------------------|
| `--auto-orient`  | Tự động xoay mặt cho đến khi trạng thái hợp lệ   |
| `--equalize`     | Bật cân bằng sáng CLAHE                            |
| `--size N`       | Kích thước chuẩn hóa ảnh mặt (mặc định: 600)      |
| `--skip-solve`   | Chỉ nhận diện màu, không giải                      |
| `--rotate-X N`   | Xoay mặt X theo N độ (0/90/180/270)                |

**Kết quả:**

```text
results/rubik/detected_cube.png         # Sơ đồ 2D trạng thái Rubik
results/rubik/recognition_details.txt   # Chi tiết nhận diện từng ô
```

### Web App (Streamlit)

Laptop và điện thoại cần cùng mạng Wi-Fi.

```bash
cd XuLyAnh

python -m streamlit run src/rubik_app.py --server.address 0.0.0.0
```

Mở trên điện thoại: `http://<IPv4-laptop>:8501`

Upload/chụp lần lượt 6 mặt theo nhãn F, U, D, L, R, B rồi bấm **"Nhận diện & Giải Rubik"**.

## Kiểm thử

```bash
cd XuLyAnh

python -m pytest tests
```

## Thuật toán nhận diện màu

1. Chia mỗi ảnh mặt thành lưới 3×3 (9 ô).
2. Lấy HSV trung bình của ô trung tâm mỗi mặt làm màu tham chiếu.
3. Với mỗi ô, tính khoảng cách HSV đến 6 màu tham chiếu (Hue xử lý theo vòng tròn 0–180).
4. Gán ô cho màu có khoảng cách nhỏ nhất.
5. Validate: mỗi màu phải xuất hiện đúng 9 lần.

**Công thức khoảng cách:**

```
ΔH = min(|H₁ - H₂|, 180 - |H₁ - H₂|)
d = √(2.0·ΔH² + 1.0·ΔS² + 0.5·ΔV²)
```

## Giới hạn hiện tại

- MVP giả định ảnh mặt Rubik đã tương đối thẳng.
- Chưa có GUI sửa màu thủ công.
- Chưa có webcam real-time.
- Chưa dùng YOLO/CNN — có thể mở rộng trong tương lai.

## Hướng phát triển

- Thêm perspective transform để xử lý ảnh nghiêng.
- Thêm GUI xác nhận và sửa màu từng ô.
- Thêm webcam capture real-time.
- Tích hợp YOLO/CNN để nhận diện mặt Rubik hoặc phân loại màu nâng cao.
- Thêm mô phỏng 3D các bước giải.

## Giấy phép

Dự án phục vụ mục đích học tập môn Xử lý Ảnh.
