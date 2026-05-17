# Rubik Cube Solver bằng xử lý ảnh

Đề tài nhận diện trạng thái Rubik 3x3 từ 6 ảnh chụp mặt cube, sau đó dùng thuật toán Kociemba để đề xuất chuỗi bước giải.

## Hướng tiếp cận

Pipeline MVP dùng OpenCV cổ điển:

```text
6 ảnh mặt Rubik
-> crop vuông trung tâm + resize
-> chia đều lưới 3x3
-> lấy màu HSV trung bình vùng giữa từng ô
-> lấy 6 ô trung tâm làm màu tham chiếu
-> gán 54 ô theo khoảng cách màu HSV có trọng số
-> validate mỗi màu xuất hiện đúng 9 lần
-> build chuỗi trạng thái theo thứ tự U R F D L B
-> gọi Kociemba solver
-> xuất sơ đồ Rubik 2D và lời giải
```

## Chuẩn bị ảnh

Đặt ảnh vào:

```text
data/rubik_samples/
```

Tên file gợi ý:

```text
U.jpg  mặt trên
R.jpg  mặt phải
F.jpg  mặt trước
D.jpg  mặt dưới
L.jpg  mặt trái
B.jpg  mặt sau
```

Lưu ý chụp ảnh:

- Chụp thẳng vuông góc từng mặt.
- Mặt Rubik chiếm phần lớn ảnh.
- Ánh sáng trắng đều, tránh bóng và lóa.
- Không xoay sai hướng các mặt khi chụp.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

## Chạy demo bằng dòng lệnh

```powershell
python src/rubik_main.py --U data/U.jpg --R data/R.jpg --F data/F.jpg --D data/D.jpg --L data/L.jpg --B data/B.jpg --output results/rubik --auto-orient
```

## Chạy web app để điện thoại chụp/upload ảnh

Laptop và điện thoại cần cùng mạng Wi-Fi.

```powershell
python -m streamlit run src/rubik_app.py --server.address 0.0.0.0
```

Lấy IPv4 laptop bằng:

```powershell
ipconfig
```

Sau đó mở trên điện thoại:

```text
http://<IPv4-laptop>:8501
```

Upload/chụp lần lượt 6 mặt theo nhãn F, U, D, L, R, B rồi bấm nhận diện và giải Rubik.

Kết quả:

```text
results/rubik/detected_cube.png
results/rubik/recognition_details.txt
```

Terminal sẽ in trạng thái 54 ký tự, trạng thái validate và chuỗi bước giải.

## Kiểm thử

```powershell
python -m pytest tests
```

## Giới hạn hiện tại

- MVP giả định ảnh mặt Rubik đã tương đối thẳng.
- Chưa có GUI sửa màu thủ công.
- Chưa có webcam real-time.
- Chưa dùng YOLO/CNN; phần này có thể làm mở rộng để cộng điểm.
