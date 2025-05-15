# Emotion Recognition from Speech

Ứng dụng web nhận diện cảm xúc qua giọng nói sử dụng Flask và TensorFlow.

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Đặt file model.h5 vào thư mục gốc của dự án

## Chạy ứng dụng

1. Kích hoạt môi trường ảo (nếu chưa kích hoạt):
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Chạy Flask server:
```bash
python app.py
```

3. Mở trình duyệt và truy cập: http://localhost:5000

## Sử dụng

1. Nhấn nút "Bắt đầu ghi âm" để bắt đầu ghi âm
2. Nói một câu để phân tích cảm xúc
3. Nhấn nút "Dừng ghi âm" để kết thúc và xem kết quả

## Lưu ý

- Đảm bảo microphone hoạt động bình thường
- Cần có kết nối internet để tải các thư viện CSS và JavaScript
- Model được train với dữ liệu tiếng Việt 