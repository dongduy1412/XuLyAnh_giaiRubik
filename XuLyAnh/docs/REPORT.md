# Báo cáo đề tài: Nhận diện màu Rubik 3x3 và đề xuất cách giải

## 1. Lý do chọn đề tài

Đề tài Rubik Solver có cấu trúc ảnh rõ ràng hơn bài toán đếm các vật thể chồng/dính nhau. Mỗi mặt Rubik là một lưới 3x3 cố định, phù hợp để áp dụng các kỹ thuật xử lý ảnh cổ điển như chuyển đổi hệ màu, tiền xử lý, phân vùng theo lưới và nhận diện màu.

## 2. Mục tiêu

- Nhận đầu vào là 6 ảnh mặt Rubik theo thứ tự U, R, F, D, L, B.
- Tách mỗi mặt thành 9 ô.
- Nhận diện màu 54 ô bằng HSV.
- Xây dựng chuỗi trạng thái Rubik chuẩn Kociemba.
- Gọi thuật toán Kociemba để đề xuất chuỗi bước giải.
- Hiển thị sơ đồ 2D của trạng thái nhận diện.

## 3. Cơ sở lý thuyết xử lý ảnh

Các kỹ thuật chính:

- Resize và crop ảnh để chuẩn hóa kích thước.
- Gaussian Blur để giảm nhiễu nhẹ.
- Chuyển đổi BGR sang HSV bằng OpenCV.
- Trích xuất màu trung bình từ vùng giữa mỗi ô.
- Tính khoảng cách màu HSV có trọng số.
- Validate trạng thái bằng ràng buộc mỗi màu xuất hiện đúng 9 lần.

## 4. Thuật toán nhận diện màu

Ý tưởng chính là dùng ô trung tâm của mỗi mặt làm màu tham chiếu. Trên Rubik 3x3, ô trung tâm không đổi vị trí, nên đại diện cho màu của mặt đó.

Quy trình:

```text
Bước 1: Chia mỗi ảnh mặt thành 9 ô.
Bước 2: Lấy HSV trung bình của ô trung tâm mỗi mặt.
Bước 3: Với từng ô, tính khoảng cách HSV đến 6 màu tham chiếu.
Bước 4: Gán ô đó cho màu có khoảng cách nhỏ nhất.
Bước 5: Kiểm tra mỗi màu xuất hiện đúng 9 lần.
```

Khoảng cách Hue xử lý theo vòng tròn 0-180 của OpenCV HSV:

```text
DeltaH = min(abs(H1 - H2), 180 - abs(H1 - H2))
```

Khoảng cách có trọng số:

```text
d = sqrt(2.0 * DeltaH^2 + 1.0 * DeltaS^2 + 0.5 * DeltaV^2)
```

## 5. Thuật toán giải Rubik

Sau khi nhận diện được 54 ô, chương trình ghép chuỗi trạng thái theo thứ tự:

```text
U R F D L B
```

Sau đó gọi thư viện Kociemba để tìm lời giải theo Two-Phase Algorithm.

## 6. Cấu trúc chương trình

```text
src/rubik_main.py
src/rubik/image_processor.py
src/rubik/face_detector.py
src/rubik/color_recognizer.py
src/rubik/rubik_state.py
src/rubik/solver.py
src/rubik/visualizer.py
```

## 7. Cách chạy

```powershell
python src/rubik_main.py --U data/rubik_samples/U.jpg --R data/rubik_samples/R.jpg --F data/rubik_samples/F.jpg --D data/rubik_samples/D.jpg --L data/rubik_samples/L.jpg --B data/rubik_samples/B.jpg --output results/rubik
```

## 8. Đánh giá

Ưu điểm:

- Bám sát nội dung xử lý ảnh cổ điển.
- Dễ demo trực quan.
- Có phần solver tạo ra kết quả ứng dụng thực tế.
- Có thể mở rộng bằng YOLO/CNN để cộng điểm.

Hạn chế:

- Ảnh phải chụp tương đối thẳng.
- Sai hướng chụp 6 mặt có thể làm trạng thái Rubik không hợp lệ.
- Ánh sáng yếu hoặc lóa có thể làm nhầm màu.
- Cần thêm giao diện cho người dùng sửa màu nếu nhận diện sai.

## 9. Hướng phát triển

- Thêm perspective transform để xử lý ảnh nghiêng.
- Thêm GUI xác nhận và sửa màu từng ô.
- Thêm webcam capture.
- Thêm YOLO/CNN để nhận diện mặt Rubik hoặc phân loại màu nâng cao.
- Thêm mô phỏng 3D các bước giải.
