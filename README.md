# Phân tích dữ liệu chất lượng không khí và xây dựng mô hình dự đoán

# Nhóm 5:
- **Trưởng nhóm**: Trần Kiều Hạnh - 22000091
- **Thành viên 1**: Đỗ Quốc An - 22000067
- **Thành viên 2**: Phạm Thị Duyên - 22000079

# Phân công công việc:
- Cả nhóm: Tiền xử lý dữ liệu, Mô hình phân loại
- Trần Kiều Hạnh: giảm chiều t-SNE, phân cụm K-means.
- Đỗ Quốc An: giảm chiều PCA, hồi quy Random Forest 
- Phạm Thị Duyên: phân cụm GMM, hồi quy MLP

## Mô tả

Thực hiện quy trình phân tích dữ liệu chất lượng không khí từ file `city_day.csv`. Quy trình bao gồm các bước tiền xử lý dữ liệu, giảm chiều dữ liệu bằng PCA và t-SNE, phân cụm dữ liệu K-means và GMM, xây dựng và đánh giá các mô hình hồi quy (Random Forest, MLP) để dự đoán chỉ số chất lượng không khí (AQI). Chuyển bài toán về bài toán phân loại rồi áp dụng mô hình Naive Bayes và mô hình phân loại Random Forest 

## Dataset
- **Lấy dữ liệu:** https://drive.google.com/file/d/1e1J6Pg28XHOpBQaH5wpblsP_I1SaqhHg/view?usp=drive_link
- **Nguồn dữ liệu:** `city_day.csv`
- **Mô tả:** Dữ liệu chất lượng không khí hàng ngày của các thành phố ở Ấn Độ từ năm 2015 đến 2020. Bao gồm các chỉ số ô nhiễm như PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene và chỉ số AQI tổng hợp.

## Báo cáo, Slide và Code
- **Link drive toàn bộ**: https://drive.google.com/drive/folders/1BZt0ouKXC2boQEg8O-3FhsrWRk4CrfYV?usp=drive_link
- **Báo cáo**: https://drive.google.com/file/d/1n5eI8x_Z_EeLOKXo7aaz6amRQ1KuBJm4/view?usp=drive_link
- **Slide**: https://drive.google.com/file/d/19_w99_BMg4kZdDdm_FqBiuyNLJQQfIBr/view?usp=drive_link
- **Mã nguồn**: https://drive.google.com/file/d/1kMyraJe_ikRYx9mTsjdMfH8fguY3Ezir/view?usp=drive_link
Note: Code có thể chạy luôn mà không cần sửa đường dẫn dữ liệu

## Thư viện cần thiết để chạy file code

Để chạy file Code_ML.ipynb, bạn cần cài đặt các thư viện Python sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Công cụ và thư viện sử dụng
- **Xử lý dữ liệu**: pandas, numpy
- **Trực quan hóa**: matplotlib, seaborn
- **Học máy**: scikit-learn
- **Giảm chiều**: PCA, t-SNE
- **Phân cụm**: K-means, Gaussian Mixture Model
- **Hồi quy**: Random Forest, MLP
- **Phân loại**: Naive Bayes, Random Forest

## Quy trình thực hiện

### 1. Tiền xử lý dữ liệu
- **Đọc và khám phá dữ liệu**: Tải dữ liệu từ file CSV, kiểm tra kích thước, kiểu dữ liệu và giá trị thiếu
- **Xử lý giá trị thiếu**: Điền giá trị thiếu bằng trung bình theo thành phố
- **Xử lý ngoại lai**: Sử dụng phương pháp IQR để phát hiện và xử lý giá trị ngoại lai
- **Chuẩn hóa dữ liệu**: Áp dụng MinMaxScaler để chuẩn hóa các biến số về thang đo 0-1
- **Tạo biến thời gian**: Tách ngày, tháng, năm từ cột Date

### 2. Phân tích thống kê mô tả
- Thống kê AQI theo năm và tháng (cả dữ liệu gốc và đã chuẩn hóa)
- Trực quan hóa xu hướng AQI qua thời gian
- Tạo heatmap thể hiện mức độ AQI theo tháng và năm
- So sánh phân phối AQI giữa dữ liệu gốc và đã chuẩn hóa

### 3. Giảm chiều dữ liệu
#### 3.1 PCA (Principal Component Analysis)
- Thực hiện PCA với toàn bộ thành phần
- Phân tích phương sai giải thích và xác định số thành phần tối ưu
- Trực quan hóa các cặp thành phần chính

#### 3.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Áp dụng t-SNE để giảm xuống 2 chiều
- Phân tích mật độ và phân bố dữ liệu
- So sánh kết quả với PCA

### 4. Phân cụm (Clustering)
#### 4.1 K-means
- Xác định số cụm tối ưu bằng Silhouette Score
- Thực hiện phân cụm trên dữ liệu gốc, PCA và t-SNE
- Đánh giá chất lượng phân cụm bằng Davies-Bouldin Index và Calinski-Harabasz Index
- Trực quan hóa kết quả phân cụm

#### 4.2 GMM (Gaussian Mixture Model)
- Sử dụng BIC để chọn số cụm và loại covariance tối ưu
- So sánh hiệu suất với K-means
- Phân tích mối quan hệ giữa các cụm và biến đầu ra AQI

### 5. Mô hình hồi quy
#### 5.1 Random Forest Regressor
- Huấn luyện với các tỉ lệ train:validation khác nhau (8:2, 7:3, 6:4)
- Thực hiện trên dữ liệu gốc, PCA, t-SNE và kết hợp với phân cụm
- Đánh giá overfitting bằng cách so sánh hiệu suất trên tập train và validation
- Phân tích Feature Importance

#### 5.2 MLP (Multi-Layer Perceptron) Regressor
- Thử nghiệm với các kích thước hidden layer khác nhau
- Áp dụng regularization để giảm overfitting
- So sánh hiệu suất với Random Forest

#### 5.3 Phân tích phần dư
- Tính toán và trực quan hóa phần dư (residuals)
- Phân tích tương quan giữa phần dư và các đặc trưng đầu vào
- So sánh phân phối phần dư giữa các cụm

### 6. Mô hình phân loại
#### 6.1 Chuẩn bị dữ liệu phân loại
- Chia AQI thành 4 khoảng có số lượng mẫu xấp xỉ nhau
- Tạo nhãn lớp: Thấp, Trung bình, Cao, Rất cao
- Giảm chiều dữ liệu xuống 1/3 số chiều ban đầu bằng PCA

#### 6.2 Naive Bayes
- Huấn luyện trên dữ liệu gốc và PCA
- Đánh giá với các metric: Accuracy, Precision, Recall, F1-score
- Vẽ ma trận nhầm lẫn (Confusion Matrix)

#### 6.3 Random Forest Classifier
- So sánh hiệu suất với Naive Bayes
- Phân tích Feature Importance
- Đánh giá trên các tỉ lệ chia dữ liệu khác nhau

## Kết quả chính
- So sánh hiệu suất các phương pháp giảm chiều
- Đánh giá tác động của phân cụm lên hiệu suất mô hình
- Phân tích overfitting và các biện pháp khắc phục
- So sánh các thuật toán học máy trên cùng một bộ dữ liệu
