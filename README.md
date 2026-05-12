# Intel Image Classification - CS114

Môn học: `CS114 - Máy học`  
Mã lớp: `CS114.Q24`  
GVHD: `Huỳnh Tân Bối`

Nhóm tác giả:
- 23521187 - Nguyễn Văn Phú
- 24520192 - Đoàn Hữu Gia Bình
- 24521163 - Trần Bảo Ngọc

## 1. Tổng quan
Đây là đồ án cuối kì, thực hiện bài toán **phân loại ảnh cảnh quan đa lớp** trên bộ dữ liệu Intel Image Classification.  
Với mỗi ảnh đầu vào, mô hình dự đoán một trong 6 lớp: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

## 2. Mục tiêu
- Xây dựng pipeline phân loại ảnh từ tiền xử lý đến đánh giá mô hình.
- So sánh các hướng tiếp cận truyền thống (ML) và học sâu (Deep Learning).
- Phân tích lỗi sai để hiểu điểm mạnh/yếu của từng mô hình.

## 3. Dataset
- Tên bộ dữ liệu: **Intel Image Classification**.
- Nguồn: Kaggle - https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- Số lớp: 6 lớp cảnh quan.
- Kích thước ảnh sau tiền xử lý: `150x150`.

Tiền xử lý chính:
- Kiểm tra ảnh lỗi.
- Phát hiện và loại ảnh trùng.
- Chia tập train/validation/test.
- Chuẩn hóa ảnh với:
  - `MEAN = [0.43018116, 0.45747542, 0.45382798]`
  - `STD = [0.26941103, 0.26793626, 0.29834034]`

## 4. Các phương pháp
Nhóm mô hình truyền thống:
- Đặc trưng: Flatten Pixel, RGB Histogram, HOG, HOG + RGB Histogram.
- Mô hình: Logistic Regression, KNN, SVM.

Nhóm học sâu:
- Simple CNN.
- ResNet18 (fine-tuning từ pretrained).

## 5. Pipeline
`Dataset -> Preprocessing -> Feature Extraction / Image Tensor -> Model Training -> Hyperparameter Tuning -> Evaluation -> Error Analysis`

## 6. Công nghệ sử dụng
- Python
- NumPy, OpenCV
- Scikit-learn
- PyTorch, torchvision
- Matplotlib
- Jupyter Notebook

## 7. Kết quả, nhận xét
Kết quả tổng quát:
- Flatten + Logistic Regression: ~36%
- Color Histogram + Logistic Regression: ~56%
- HOG + Logistic Regression: ~72%
- HOG + RGB Histogram + SVM: ~79%
- Simple CNN: ~87%
- ResNet18 fine-tuned: ~92-93%

Nhận xét:
- HOG hiệu quả với thông tin cạnh/texture, nhưng còn hạn chế ngữ nghĩa.
- Kết hợp HOG + RGB giúp cải thiện rõ rệt so với baseline ML.
- Mô hình deep learning vượt trội hơn các đặc trưng thủ công.
- Các cặp dễ nhầm: `glacier <-> mountain`, `buildings <-> street`, `sea <-> mountain`.

## 8. Hướng phát triển
- Thử nghiệm thêm các kiến trúc mạnh hơn như EfficientNet, ConvNeXt, ViT.
- Tăng cường kỹ thuật data augmentation để cải thiện khả năng tổng quát hóa.
- Mở rộng phân tích giải thích mô hình bằng Grad-CAM/Explainable AI.
- Tối ưu mô hình cho suy luận nhanh và triển khai thực tế.
