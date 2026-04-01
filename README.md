# ĐỒ ÁN MÔN MÁY HỌC - CS114.Q24

**Giảng viên:** Huỳnh Tân Bối  
**Đề tài:** [Tên đề tài]  
**Nhóm:** [Tên nhóm]  

## Danh sách sinh viên thực hiện

| STT | MSSV | Họ tên |
|-----|------|--------|
| 1 | [MSSV 1] | [Họ tên 1] |
| 2 | [MSSV 2] | [Họ tên 2] |
| 3 | [MSSV 3] | [Họ tên 3] |

---

# I. Giới thiệu đồ án

## 1. Giới thiệu
Trong những năm gần đây, Học máy (Machine Learning) và Học sâu (Deep Learning) đã trở thành những lĩnh vực quan trọng trong khoa học dữ liệu và trí tuệ nhân tạo. Một trong những bài toán cơ bản và phổ biến trong lĩnh vực thị giác máy tính (Computer Vision) là bài toán phân loại hình ảnh (Image Classification). Mục tiêu của bài toán này là xây dựng mô hình có khả năng tự động phân loại hình ảnh vào các lớp khác nhau dựa trên các đặc trưng trích xuất từ dữ liệu ảnh.

Phân loại cảnh vật (Scene Classification) là một dạng của bài toán phân loại hình ảnh, trong đó mô hình cần phân biệt các loại cảnh khác nhau như rừng, núi, biển, khu đô thị hay đường phố. Bài toán này có độ khó tương đối cao do các lớp có thể có nhiều đặc trưng tương đồng về màu sắc và kết cấu, ví dụ như cảnh núi và sông băng, hoặc khu đô thị và đường phố. Do đó, việc lựa chọn mô hình và phương pháp trích xuất đặc trưng phù hợp là rất quan trọng.

Trong đồ án này, nhóm thực hiện xây dựng một mô hình học máy nhằm giải quyết bài toán phân loại cảnh vật sử dụng dữ liệu hình ảnh. Quy trình thực hiện bao gồm các bước chính như: phân tích dữ liệu (EDA), tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện mô hình và đánh giá kết quả. Thông qua đồ án này, nhóm mong muốn hiểu rõ hơn về quy trình xây dựng một mô hình học máy hoàn chỉnh cho bài toán phân loại hình ảnh.

---

## 2. Giới thiệu Dataset
Bộ dữ liệu được sử dụng trong đồ án là Intel Image Classification Dataset. Đây là bộ dữ liệu bao gồm các hình ảnh về nhiều loại cảnh khác nhau trong tự nhiên và môi trường đô thị, thường được sử dụng trong các bài toán phân loại cảnh vật trong lĩnh vực thị giác máy tính.

Dataset bao gồm 6 lớp cảnh:
- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Tập dữ liệu được chia thành hai phần:
- Tập huấn luyện (Training set): dùng để huấn luyện mô hình
- Tập kiểm tra (Test set): dùng để đánh giá mô hình

Các hình ảnh trong dataset là ảnh màu RGB và đã được chuẩn hóa về cùng kích thước. Mỗi lớp dữ liệu có các đặc trưng khác nhau về màu sắc, kết cấu và bố cục hình ảnh. Ví dụ, lớp Forest thường có nhiều màu xanh lá, lớp Sea có màu xanh dương là chủ đạo, trong khi lớp Buildings và Street chứa nhiều cấu trúc nhân tạo như tòa nhà và đường phố.

---

## 3. Mục tiêu và phạm vi đồ án

### Mục tiêu
Mục tiêu của đồ án này bao gồm:
- Tìm hiểu bài toán phân loại hình ảnh và các phương pháp học máy sử dụng trong bài toán này
- Thực hiện phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA)
- Thực hiện tiền xử lý dữ liệu và tăng cường dữ liệu
- Xây dựng mô hình học máy/học sâu để phân loại cảnh vật
- Đánh giá hiệu suất của mô hình thông qua các độ đo đánh giá
- Phân tích các trường hợp mô hình dự đoán sai

### Phạm vi
Trong đồ án này, nhóm tập trung vào:
- Sử dụng dataset Intel Image Classification
- Áp dụng các mô hình học sâu như CNN hoặc Transfer Learning
- Đánh giá mô hình dựa trên độ chính xác và ma trận nhầm lẫn (Confusion Matrix)

Đồ án không tập trung vào việc tối ưu hóa mô hình ở mức cao mà tập trung vào việc hiểu và triển khai quy trình xây dựng mô hình học máy.

---

## 4. Ý nghĩa của đề tài
Bài toán phân loại cảnh vật có nhiều ứng dụng trong thực tế như:
- Nhận diện môi trường trong xe tự lái
- Hệ thống tìm kiếm và phân loại hình ảnh tự động
- Giám sát môi trường thông qua ảnh vệ tinh
- Robot và drone tự hành
- Hệ thống gợi ý địa điểm du lịch

Thông qua việc thực hiện đồ án này, sinh viên có thể hiểu rõ quy trình xây dựng một hệ thống học máy hoàn chỉnh, từ bước phân tích dữ liệu, tiền xử lý dữ liệu, xây dựng mô hình, huấn luyện mô hình cho đến đánh giá và phân tích kết quả.

---
# 2. Tổng quan EDA (Exploratory Data Analysis)

## 1. Tổng quan Dataset
- **Tập huấn luyện:** 14,034 ảnh (128×128 RGB)
- **Tập kiểm tra:** 3,000 ảnh (128×128 RGB)
- **Số lớp:** 6 loại cảnh vật - Buildings, Forest, Glacier, Mountain, Sea, Street
- **Phân bố lớp:** Dữ liệu được phân bố tương đối đồng đều giữa các lớp

---

## 2. Chất lượng dữ liệu ✓
- **Ảnh bị mờ:** 19 ảnh (0.14%) → Chất lượng dữ liệu rất tốt
- **Giá trị thiếu:** Không phát hiện
- **Ảnh sai kích thước:** Không có
- **Outliers:** 19 ảnh (0.14%) có phân bố pixel bất thường, chủ yếu do ánh sáng quá mạnh hoặc bố cục ảnh đặc biệt

→ Kết luận: Dataset sạch, hầu như không cần làm sạch dữ liệu.

---

## 3. Đặc trưng hình ảnh
- **Chuẩn hóa pixel:** Đã đưa về khoảng [0, 1]
- **Phân bố màu RGB khác nhau theo từng lớp:**
  - **Glacier:** Kênh Blue cao nhất (0.591) → đặc trưng băng, tuyết
  - **Forest:** Kênh Blue thấp nhất (0.249) → đặc trưng cây cối (màu xanh lá)
  - **Các lớp khác:** RGB khá cân bằng (0.41 – 0.50)

→ Màu sắc là một đặc trưng quan trọng để phân loại cảnh vật.

---

## 4. Khả năng phân tách đặc trưng (PCA Analysis)
- **Phương sai PC1:** 24.11%
- **Phương sai PC2:** 13.80%
- **Tổng phương sai trong không gian 2D:** 37.91%
- **Mức độ phân tách lớp:** Các lớp có xu hướng tạo cụm nhưng vẫn chồng lấn đáng kể

→ Điều này cho thấy bài toán không thể phân loại tốt bằng phương pháp đơn giản → cần mô hình CNN để trích xuất đặc trưng sâu.

---

## 5. Phân tích tương quan giữa các lớp
- **Tương quan cao:**
  - Mountain ↔ Buildings (0.974)
  - Mountain ↔ Sea (0.974)
  - Buildings ↔ Sea (0.959)
  
→ Các lớp này có đặc trưng hình ảnh tương đối giống nhau → dễ bị nhầm lẫn khi phân loại.

- **Tương quan thấp:**
  - Forest ↔ Glacier (0.242) → Hai lớp dễ phân biệt nhất

→ Một số lớp sẽ khó phân loại hơn và cần mô hình học đặc trưng mạnh.

---

## 6. Kết luận & Đề xuất
✅ **Dataset phù hợp để huấn luyện mô hình**
- Dữ liệu sạch, ít outliers (0.14%)
- Phân bố lớp cân bằng → không cần resampling
- Nên sử dụng **Data Augmentation** (xoay, lật, zoom, thay đổi độ sáng) để tăng khả năng tổng quát hóa
- Nên sử dụng mô hình CNN sâu như **ResNet, EfficientNet**
- Cần chú ý các cặp lớp dễ nhầm lẫn: **Mountain – Buildings – Sea**

---

## 7. Tổng kết EDA
EDA cho thấy dataset có chất lượng tốt, các lớp có sự khác biệt về màu sắc và đặc trưng hình ảnh, tuy nhiên vẫn có sự chồng lấn giữa một số lớp. Do đó, việc sử dụng các mô hình học sâu (Deep Learning) là cần thiết để trích xuất đặc trưng và đạt độ chính xác cao trong bài toán phân loại cảnh vật.