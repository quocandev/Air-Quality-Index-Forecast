# Dự án Phân tích Dữ liệu Chất lượng Không khí và Xây dựng Mô hình Dự đoán

**Nhóm 5**: Trần Kiều Hạnh, Đỗ Quốc An, Phạm Thị Duyên

## Mô tả Dự án

Dự án này thực hiện phân tích toàn diện dữ liệu chất lượng không khí tại các thành phố của Ấn Độ và xây dựng các mô hình machine learning để dự đoán chỉ số AQI (Air Quality Index).

## Mục tiêu

1. **Phân tích khám phá dữ liệu (EDA)**: Hiểu rõ đặc điểm và xu hướng của dữ liệu chất lượng không khí
2. **Tiền xử lý dữ liệu**: Làm sạch và chuẩn bị dữ liệu cho mô hình
3. **Giảm chiều dữ liệu**: Áp dụng PCA và t-SNE để giảm chiều và trực quan hóa
4. **Phân cụm**: Sử dụng K-means và GMM để nhóm các quan sát
5. **Mô hình hồi quy**: Dự đoán giá trị AQI liên tục
6. **Mô hình phân loại**: Phân loại mức độ chất lượng không khí

## Cấu trúc Dự án

```
Air-Quality-Index-Forecast/
├── LICENSE                     # Giấy phép
├── main.py                     # File chạy chính
├── README.md                   # Tài liệu dự án
├── requirements.txt            # Thư viện cần thiết
├── __init__.py                # Package init
├── config/                     # Cấu hình
│   ├── __init__.py
│   └── config.py
├── data/                       # Dữ liệu
│   ├── __init__.py
│   ├── data_loader.py          # Load dữ liệu
│   ├── preprocessing.py        # Tiền xử lý
│   └── raw/
│       ├── __init__.py
│       └── city_day.csv        # Dữ liệu thô
├── doc/                        # Tài liệu
│   ├── __init__.py
│   ├── report.pdf              # Báo cáo
│   └── slide.pdf               # Slide thuyết trình
├── output/                     # Kết quả
│   ├── __init__.py
│   └── tien-xu-ly.txt         # Ghi chú tiền xử lý
└── src/                        # Mã nguồn chính
    ├── __init__.py
    ├── Code_ML.ipynb           # Jupyter notebook
    ├── features/               # Xử lý đặc trưng
    │   ├── __init__.py
    │   └── demensionality_reduction.py
    ├── models/                 # Mô hình ML
    │   ├── __init__.py
    │   ├── classification.py
    │   ├── clustering.py
    │   └── regression.py
    └── visualization/          # Trực quan hóa
        ├── __init__.py
        └── eda_plots.py
```

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/quocandev/Air-Quality-Index-Forecast.git
cd Air-Quality-Index-Forecast
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy toàn bộ pipeline

```bash
python main.py
```

### Chạy từng module riêng biệt

```python
from config.config import Config
from data.data_loader import DataLoader
from src.models.clustering import ClusteringModels

# Load cấu hình
config = Config()

# Load dữ liệu
data_loader = DataLoader(config)
df = data_loader.load_data()

# Thực hiện phân cụm
clustering = ClusteringModels(config)
results = clustering.perform_clustering(df, X_pca, X_tsne)
```

## Dữ liệu

### Mô tả dataset
- **Nguồn**: Dữ liệu chất lượng không khí tại các thành phố Ấn Độ
- **Kích thước**: ~29,000 quan sát, 16 features
- **Thời gian**: 2015-2020

### Các features chính
- **Pollutants**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target**: AQI (Air Quality Index)
- **Location**: City (thành phố)
- **Time**: Date, Year, Month, Day
- **Category**: AQI_Bucket

## Phương pháp

### 1. Tiền xử lý dữ liệu
- Xử lý giá trị thiếu
- Phát hiện và xử lý outliers
- Chuẩn hóa dữ liệu
- Tạo biến dummy

### 2. Giảm chiều dữ liệu
- **PCA**: Giảm chiều tuyến tính
- **t-SNE**: Giảm chiều phi tuyến cho trực quan hóa

### 3. Phân cụm
- **K-means**: Phân cụm cứng
- **Gaussian Mixture Model**: Phân cụm mềm

### 4. Mô hình dự đoán
- **Random Forest Regressor**: Dự đoán AQI liên tục
- **MLP Regressor**: Mạng neural cho hồi quy
- **Naive Bayes**: Phân loại mức độ AQI
- **Random Forest Classifier**: Phân loại robust

## Kết quả

### Metrics đánh giá
- **Hồi quy**: MSE, RMSE, R², MAE
- **Phân loại**: Accuracy, Precision, Recall, F1-Score
- **Phân cụm**: Silhouette Score, Davies-Bouldin Index

### Kết quả chính
- Mô hình Random Forest đạt hiệu suất tốt nhất
- PCA giữ lại 95% phương sai với 8 components
- K-means với k=4 phù hợp nhất cho dữ liệu

## Cấu trúc File Output

```
output/
├── project.log                 # Log chạy chương trình
├── summary_statistics.csv      # Thống kê tóm tắt
├── regression_results.csv      # Kết quả mô hình hồi quy
├── classification_results.csv  # Kết quả mô hình phân loại
├── aqi_distribution.png        # Biểu đồ phân phối AQI
├── correlation_matrix.png      # Ma trận tương quan
├── city_comparison.png         # So sánh thành phố
├── temporal_analysis.png       # Phân tích thời gian
├── pca_variance.png           # Phương sai PCA
├── pca_tsne_comparison.png    # So sánh PCA và t-SNE
├── clustering_visualization.png # Trực quan hóa phân cụm
├── regression_comparison.png   # So sánh mô hình hồi quy
├── classification_comparison.png # So sánh mô hình phân loại
└── confusion_matrices.png     # Ma trận nhầm lẫn
```

## Yêu cầu Hệ thống

- Python 3.8+
- RAM: 4GB+ (khuyến nghị 8GB)
- Disk space: 2GB+

## Giấy phép

Dự án này được phát hành dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## Liên hệ

- Email: doquocan_t67@hus.edu.vn
- GitHub: https://github.com/quocandev