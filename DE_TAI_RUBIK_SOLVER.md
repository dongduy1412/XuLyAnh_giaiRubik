# Đề tài: Rubik Cube Image Solver bằng xử lý ảnh

## 1. Mục tiêu đề tài
Mục tiêu của đề tài là xây dựng một ứng dụng nhận diện và giải Rubik 3x3 từ ảnh thật. Người dùng có thể:

- chụp hoặc tải lên ảnh từng mặt Rubik,
- hệ thống tự tách mặt Rubik ra khỏi ảnh,
- chia mặt thành lưới 3x3,
- nhận diện màu từng ô,
- ghép lại thành trạng thái Rubik chuẩn,
- kiểm tra tính hợp lệ và sinh lời giải bằng thư viện `kociemba`.

Ứng dụng được làm theo hướng trực quan, dễ dùng, phù hợp để demo đồ án môn Xử lý ảnh.

---

## 2. Ý tưởng và hướng triển khai
Đề tài được chia thành 4 phần chính:

1. **Tiền xử lý ảnh**
   - đọc ảnh từ file hoặc camera,
   - chỉnh ảnh theo EXIF,
   - crop vùng trung tâm hoặc tự phát hiện vùng mặt Rubik,
   - chuẩn hóa kích thước và ánh sáng.

2. **Tách lưới và nhận diện màu**
   - chia mặt Rubik thành 9 ô,
   - lấy màu trung bình của từng ô,
   - so sánh với bộ màu tham chiếu,
   - gán nhãn cho từng ô theo 6 màu của Rubik.

3. **Ghép trạng thái Rubik**
   - gom nhãn từ 6 mặt,
   - tạo chuỗi trạng thái 54 ký tự theo chuẩn Kociemba,
   - kiểm tra tính hợp lệ của cube state.

4. **Giải Rubik**
   - dùng thư viện `kociemba` để sinh lời giải,
   - hiển thị kết quả giải và layout Rubik đã nhận diện.

---

## 3. Kế hoạch thực hiện
Kế hoạch ban đầu của dự án gồm các bước sau:

- tạo cấu trúc project rõ ràng,
- port lại các module xử lý ảnh từ project cũ,
- giữ thư viện `kociemba` dạng vendored trong repo,
- xây dựng file cấu hình màu tham chiếu từ thư mục `Color/`,
- làm UI Streamlit để người dùng thao tác từng mặt Rubik,
- thêm CLI để có thể chạy bằng dòng lệnh,
- viết test cho các phần lõi như nhận diện màu, state string và solver.

---

## 4. Những gì đã làm được
Hiện tại project đã có các phần sau:

### 4.1. Core xử lý ảnh
- đọc ảnh và chuẩn hóa ảnh đầu vào,
- crop vùng chứa mặt Rubik,
- chia mặt Rubik thành 3x3,
- lấy màu trung bình trong từng ô,
- nhận diện màu bằng không gian Lab,
- có cơ chế hiệu chỉnh bằng màu tham chiếu.

### 4.2. Profile màu
- dùng 6 ảnh trong thư mục `Color/` để dựng profile ban đầu,
- lưu cache vào `data/rubik_color_profile.json`,
- có thể cập nhật theo các mặt đã nhận diện trong layout.

### 4.3. UI Streamlit
- giao diện dạng layout Rubik 6 mặt,
- bấm `+` để thêm từng mặt,
- upload ảnh hoặc chụp camera,
- xem ảnh crop + lưới 3x3,
- sửa nhãn nếu nhận diện sai,
- khi đủ 6 mặt thì kiểm tra và giải Rubik.

### 4.4. CLI
- có file chạy dòng lệnh riêng để test nhanh,
- dùng cùng pipeline với UI,
- có thể chạy trên từng ảnh từng mặt.

### 4.5. Test
Đã có test cho:
- profile màu,
- nhận diện màu,
- state string của Rubik,
- wrapper solver.

---

## 5. Cấu trúc thư mục chính

```text
XuLyAnh1/
├─ Color/                    # Ảnh mẫu tham chiếu màu
├─ data/
│  ├─ B.jpg ... U.jpg         # Ảnh mặt Rubik đã lưu
│  └─ rubik_color_profile.json # Cache profile màu
├─ kociemba/                 # Thư viện giải Rubik vendored
├─ src/
│  ├─ rubik_app.py            # Ứng dụng Streamlit
│  ├─ rubik_main.py           # Chạy CLI
│  └─ rubik/                  # Các module core
├─ tests/                     # Unit test
├─ requirements.txt           # Thư viện cần cài
└─ DE_TAI_RUBIK_SOLVER.md     # File mô tả đề tài này
```

---

## 6. Cách setup đơn giản

### Bước 1: Cài Python
Nên dùng **Python 3.12** hoặc **3.13** trên Windows để ổn định hơn. Nếu đang dùng Python 3.14 mà gặp lỗi mạng/sockets khi chạy Streamlit, nên đổi sang Python 3.12.

### Bước 2: Mở PowerShell và vào thư mục project
```powershell
cd F:\XLA_demdoituong\XuLyAnh1
```

### Bước 3: Tạo môi trường ảo
```powershell
python -m venv .venv
```

### Bước 4: Kích hoạt môi trường ảo
```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chạy một lần:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Bước 5: Cài thư viện
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 6: Chạy ứng dụng web
```powershell
python -m streamlit run src\rubik_app.py --server.address 127.0.0.1 --server.port 8501
```

Mở trình duyệt tại:
```text
http://127.0.0.1:8501
```

### Bước 7: Chạy bản dòng lệnh
Ví dụ:
```powershell
python src\rubik_main.py --help
```

---

## 7. Cách dùng nhanh trong app

1. Mở app Streamlit.
2. Bấm dấu `+` ở mặt cần thêm.
3. Chọn upload ảnh hoặc chụp ảnh.
4. Bấm nhận diện màu.
5. Kiểm tra lưới 3x3.
6. Nếu nhận diện sai, sửa trực tiếp từng ô.
7. Thêm đủ 6 mặt.
8. Bấm kiểm tra và giải Rubik.

---

## 8. Lưu ý khi chạy
- Thư mục `Color/` là bộ ảnh tham chiếu màu ban đầu.
- File `data/rubik_color_profile.json` là cache profile màu, app có thể tự cập nhật lại.
- Thư viện `kociemba` đã được để sẵn trong repo, không cần cài riêng.
- Không nên commit các file `__pycache__/`.

---

## 9. Ghi chú cho các thành viên trong nhóm
Nếu muốn test nhanh, chỉ cần:

```powershell
cd F:\XLA_demdoituong\XuLyAnh1
.\.venv\Scripts\Activate.ps1
python -m streamlit run src\rubik_app.py
```

Nếu app nhận diện sai màu, nên:
- chụp ảnh rõ hơn,
- để mặt Rubik chiếm nhiều khung hình hơn,
- kiểm tra ánh sáng,
- sửa nhãn trực tiếp trên UI,
- mở lại app để profile màu được cập nhật.

---

## 10. Tóm tắt ngắn
Đây là đồ án xử lý ảnh nhận diện Rubik 3x3 từ ảnh thật. Project đã có UI web, CLI, pipeline nhận diện, profile màu và solver. Người dùng chỉ cần cài Python, cài thư viện, chạy Streamlit là có thể test ngay.
