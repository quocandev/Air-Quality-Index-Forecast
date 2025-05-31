# Phân tích dữ liệu chất lượng không khí và xây dựng mô hình dự đoán

# Nhóm 5:
- **Trưởng nhóm**: Trần Kiều Hạnh - 22000091
- **Thành viên 1**: Đỗ Quốc An - 22000067
- **Thành viên 2**: Phạm Thị Duyên - 22000079

## Mô tả

Thực hiện quy trình phân tích dữ liệu chất lượng không khí từ file `city_day.csv`. Quy trình bao gồm các bước tiền xử lý dữ liệu, giảm chiều dữ liệu bằng PCA và t-SNE, xây dựng và đánh giá các mô hình hồi quy (Random Forest, MLP) để dự đoán chỉ số chất lượng không khí (AQI).

Mã nguồn `code.ipynb` dùng để so sánh các trường hợp khác nhau của Random Forest và MLP đồng thời đánh giá overfit. Trong khi mã nguồn `code2.ipynb` dùng để đánh giá phần dư của hai mô hình

## Dataset
- **Lấy dữ liệu:** Trước tiên ta cần download file dữ liệu city_day.csv tại link google drive: https://drive.google.com/file/d/1-R9pNlc7VY-78eZK9X12uMn7oshDwfGV/view?usp=drive_link
- Sau đó, mở file code2.ipynb ở phần đầu tiền xử lý dữ liệu df = pd.read_csv(“...”)  hãy chèn đường dẫn đến file dữ liệu vừa tải
- **Nguồn dữ liệu:** `city_day.csv`
- **Mô tả:** Dữ liệu chất lượng không khí hàng ngày của các thành phố ở Ấn Độ từ năm 2015 đến 2020. Bao gồm các chỉ số ô nhiễm như PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene và chỉ số AQI tổng hợp.

## Báo cáo, Slide và Code
- **Link drive toàn bộ**: https://drive.google.com/drive/folders/12YnMScIMv5_1CYoJe1vjnFw2u_Fztass?usp=drive_link
- **Báo cáo**: https://drive.google.com/file/d/19tIzBQXdn1VcPhqaLEB6xFS_aFRdwMno/view?usp=sharing
- **Slide**: https://drive.google.com/file/d/1lcsIeRYTrKzezkgil28ryKbUTfYdkk97/view?usp=sharing
- **Mã nguồn code.ipynb**: https://drive.google.com/file/d/1r_p0N0HXrr8AnoM7tgnRZL6JcAzXcN7R/view?usp=sharing
- **Mã nguồn code2.ipynb**: https://drive.google.com/file/d/18Wfa6jYh9NrleMfFQpGoy3duOsUoAUhO/view?usp=sharing

## Thư viện cần thiết để chạy file code

Để chạy file code.ipynb và code2.ipynb, bạn cần cài đặt các thư viện Python sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Các bước thực hiện trong file 

1.  **Tải và khám phá dữ liệu:**
    *   Đọc dữ liệu từ file CSV.
    *   Hiển thị thông tin cơ bản, thống kê mô tả, kiểm tra giá trị thiếu.

2.  **Tiền xử lý dữ liệu:**
    *   Chuyển đổi cột `Date` sang định dạng datetime.
    *   Thêm các cột `Year`, `Month`, `Day`.
    *   Xử lý giá trị thiếu cho các cột số (sử dụng giá trị trung bình theo thành phố và trung bình toàn cục).
    *   Xử lý giá trị thiếu cho cột `AQI_Bucket` (ánh xạ từ giá trị `AQI`).
    *   Phát hiện và xử lý giá trị ngoại lai (sử dụng IQR clipping).
    *   Chuyển đổi biến hạng mục (`AQI_Bucket`, `City`) sang dạng one-hot encoding.
    *   Chuẩn hóa các biến số (`MinMaxScaler`).

3.  **Giảm chiều dữ liệu với PCA:**
    *   Áp dụng PCA để giữ lại 95% phương sai của dữ liệu.
    *   Phân tích chi tiết các thành phần chính (phương sai giải thích, trọng số).
    *   Trực quan hóa dữ liệu sau khi giảm chiều (2D, 3D).
    *   Phân tích mối quan hệ giữa các thành phần chính và AQI.

4.  **Giảm chiều và trực quan hóa với t-SNE:**
    *   Áp dụng t-SNE để giảm chiều dữ liệu xuống 2 và 3 chiều.
    *   Trực quan hóa kết quả t-SNE, tô màu theo `AQI_Bucket`.
    *   Phân tích cấu trúc cụm và mật độ dữ liệu trong không gian t-SNE (sử dụng K-Means).
    *   Phân tích mối quan hệ giữa biểu diễn t-SNE và AQI.

5.  **So sánh hiệu suất dự đoán (PCA vs t-SNE vs Dữ liệu gốc):**
    *   Huấn luyện mô hình Random Forest trên dữ liệu gốc đã chuẩn hóa, dữ liệu PCA và dữ liệu t-SNE.
    *   Đánh giá và so sánh hiệu suất (RMSE, MAE, R²) của các mô hình.

6.  **So sánh đặc điểm PCA và t-SNE:**
    *   Trực quan hóa song song kết quả PCA và t-SNE.
    *   Tổng hợp bảng so sánh các đặc điểm chính của hai phương pháp.

7.  **Huấn luyện và đánh giá Random Forest với các tỉ lệ phân chia:**
    *   Thực nghiệm với các tỉ lệ train/validation khác nhau (8:2, 7:3, 6:4) trên dữ liệu gốc, dữ liệu PCA và dữ liệu t-SNE.
    *   Phân tích kết quả, đánh giá hiệu suất và tầm quan trọng của các đặc trưng.

8.  **Phân tích phần dư (Residual Analysis) cho Random Forest:**
    *   Phân tích phân phối phần dư, tương quan với biến đầu vào, và sự thay đổi theo giá trị dự đoán/thực tế.
    *   Đánh giá tính phù hợp của mô hình Random Forest.

9.  **Huấn luyện và đánh giá MLP với các tỉ lệ phân chia:**
    *   Thực nghiệm với các tỉ lệ train/validation khác nhau trên dữ liệu gốc đã chuẩn hóa, dữ liệu PCA và dữ liệu t-SNE.
    *   Phân tích đường cong học (Learning Curve).
    *   Phân tích phần dư cho mô hình MLP tốt nhất.

10. **So sánh MLP và Random Forest:**
    *   Tổng hợp và so sánh hiệu suất, thời gian huấn luyện của hai mô hình trên các cấu hình khác nhau.

11. **Kết luận:**
    *   Đưa ra kết luận về hiệu quả của các phương pháp giảm chiều và mô hình hồi quy.
    *   Đề xuất mô hình và cấu hình tốt nhất cho bài toán.

## Cách chạy

1.  Đảm bảo bạn đã cài đặt các thư viện cần thiết (xem mục **Thư viện cần thiết**).
2.  Mở notebook `code.ipynb` và `code2.ipynb` bằng Jupyter Notebook, JupyterLab, VS Code hoặc môi trường tương thích khác.
3.  Chạy tuần tự các cell code từ trên xuống dưới.

## Kết quả chính

- Việc áp dụng PCA giúp cải thiện đáng kể hiệu suất của cả mô hình Random Forest và MLP, đồng thời giảm thời gian huấn luyện.
- Mô hình Random Forest cho kết quả dự đoán tốt hơn một chút so với MLP trong các thử nghiệm.
- Tỉ lệ phân chia train:validation 7:3 cho kết quả tối ưu trên cả hai mô hình với dữ liệu PCA.
- Phân tích phần dư cho thấy cả hai mô hình đều phù hợp với bài toán, không có thiên lệch hệ thống rõ ràng.
