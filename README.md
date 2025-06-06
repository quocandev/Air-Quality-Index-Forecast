# Air Quality Index Analysis and Prediction

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Nhóm 5**: Trần Kiều Hạnh, Đỗ Quốc An, Phạm Thị Duyên  
> **Môn học**: Học Máy  
> **Giảng viên**: Cao Văn Chung

## 📋 Mô tả Dự án

Dự án này thực hiện phân tích toàn diện dữ liệu chất lượng không khí tại 26 thành phố của Ấn Độ (2015-2020) và xây dựng các mô hình machine learning để dự đoán chỉ số AQI (Air Quality Index). Dự án áp dụng đầy đủ quy trình Data Science từ EDA, tiền xử lý dữ liệu, giảm chiều, phân cụm đến xây dựng mô hình dự đoán.

## 🎯 Mục tiêu

- [x] **Khám phá dữ liệu (EDA)**: Phân tích đặc điểm và xu hướng chất lượng không khí
- [x] **Tiền xử lý dữ liệu**: Xử lý missing values, outliers, chuẩn hóa dữ liệu  
- [x] **Giảm chiều dữ liệu**: Áp dụng PCA và t-SNE cho visualization
- [x] **Phân cụm**: K-means và Gaussian Mixture Model
- [x] **Mô hình hồi quy**: Random Forest và MLP Regressor
- [x] **Mô hình phân loại**: Naive Bayes và Random Forest Classifier

## 📊 Dữ liệu

### Dataset Overview
- **Nguồn**: Air Quality Data in India (2015-2020)
- **Kích thước**: 29,531 quan sát × 16 features
- **Phạm vi**: 26 thành phố Ấn Độ
- **Thời gian**: 2015-01-01 đến 2020-07-01

### Features
```python
# Pollutants (12 features)
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
                  'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Target variable
TARGET_COL = 'AQI'  # Air Quality Index

# Categorical features  
CATEGORICAL_COLS = ['City', 'AQI_Bucket']

# Time features
TIME_COLS = ['Date', 'Year', 'Month', 'Day']
```

## 🏗️ Cấu trúc Dự án

```
nhom5_AnDuyenHanh/
├── 📄 README.md                    # Tài liệu dự án
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Dependencies
├── 🐍 main.py                      # Entry point
├── 🐍 __init__.py                  # Package marker
│
├── ⚙️ config/                      # Cấu hình
│   ├── __init__.py
│   └── config.py                   # Config class
│
├── 📁 data/                        # Data handling
│   ├── __init__.py
│   ├── data_loader.py              # DataLoader class  
│   ├── preprocessing.py            # DataPreprocessor class
│   └── raw/
│       ├── __init__.py
│       └── city_day.csv            # Raw dataset
│
├── 📁 src/                         # Source code
│   ├── __init__.py
│   ├── 📓 Code_ML.ipynb            # Jupyter notebook
│   ├── features/
│   │   ├── __init__.py
│   │   └── dimensionality_reduction.py  # PCA, t-SNE
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py           # K-means, GMM
│   │   ├── regression.py           # RF, MLP Regressor
│   │   └── classification.py       # NB, RF Classifier
│   └── visualization/
│       ├── __init__.py
│       └── eda_plots.py            # EDA visualizations
│
├── 📁 utils/                       # Utilities
│   └── font_config.py              # Matplotlib font setup
│
├── 📁 output/                      # Results
│   ├── __init__.py
│   ├── 📊 *.png                    # Plots
│   ├── 📊 *.csv                    # Results
│   └── 📄 project.log              # Execution logs
│
└── 📁 doc/                         # Documentation
    ├── __init__.py
    ├── report.pdf                  # Technical report
    └── slide.pdf                   # Presentation
```

## 🚀 Cài đặt và Sử dụng

### Prerequisites
- Python 3.8+
- RAM: 4GB+ (khuyến nghị 8GB cho t-SNE)
- Disk space: 2GB+

### Installation

```bash
# 1. Clone repository
git clone https://github.com/quocandev/Air-Quality-Index-Forecast.git
cd Air-Quality-Index-Forecast

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### Usage

#### Chạy toàn bộ pipeline
```bash
python main.py
```

#### Chạy từng module riêng biệt
```python
from config.config import Config
from data.data_loader import DataLoader
from data.preprocessing import DataPreprocessor
from src.models.clustering import ClusteringModels

# Load configuration
config = Config()

# Load và preprocess data
data_loader = DataLoader(config)
df = data_loader.load_data()

preprocessor = DataPreprocessor(config)
df_processed = preprocessor.preprocess(df)

# Thực hiện phân cụm
clustering = ClusteringModels(config)
results = clustering.perform_clustering(df_processed, X_pca, X_tsne)
```

## 🔬 Phương pháp

### 1. Data Preprocessing
- **Missing values**: Forward fill + interpolation
- **Outliers**: IQR method với clipping
- **Feature engineering**: One-hot encoding cho categorical
- **Normalization**: MinMaxScaler (0-1) và StandardScaler

### 2. Dimensionality Reduction
- **PCA**: 10 components giữ lại 95% variance
- **t-SNE**: 2D visualization với perplexity=30, learning_rate=200

### 3. Clustering
- **K-means**: Optimal k=4 clusters
- **Gaussian Mixture Model**: Soft clustering với BIC selection

### 4. Machine Learning Models

#### Regression (AQI prediction)
- **Random Forest Regressor**: n_estimators=[100, 200]
- **MLP Regressor**: hidden_layers=[100, 50]

#### Classification (AQI_Bucket prediction)  
- **Naive Bayes**: Gaussian NB
- **Random Forest Classifier**: n_estimators=[100, 200]

### 5. Model Evaluation
- **Regression**: MSE, RMSE, R², MAE
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Clustering**: Silhouette Score, Davies-Bouldin Index

## 📈 Kết quả

### Model Performance
| Model | Type | Best R²/Accuracy | RMSE/F1-Score |
|-------|------|------------------|---------------|
| Random Forest | Regression | 0.95 | 15.2 |
| MLP | Regression | 0.92 | 18.7 |
| Random Forest | Classification | 0.94 | 0.93 |
| Naive Bayes | Classification | 0.87 | 0.85 |

### Key Findings
- **Best Model**: Random Forest (cả regression và classification)
- **PCA**: 10 components giữ 95% variance
- **Clustering**: K=4 optimal cho dữ liệu
- **Feature Importance**: PM2.5, PM10, NO2 là factors quan trọng nhất

## 📁 Output Files

```
output/
├── 📄 project.log                      # Execution logs
├── 📊 summary_statistics.csv           # Dataset statistics
├── 📊 regression_results.csv           # Regression metrics
├── 📊 classification_results.csv       # Classification metrics
├── 📈 aqi_distribution.png             # AQI distribution
├── 📈 correlation_matrix.png           # Feature correlations
├── 📈 city_comparison.png              # City-wise analysis
├── 📈 temporal_analysis.png            # Time series analysis
├── 📈 pca_variance.png                 # PCA explained variance
├── 📈 pca_tsne_comparison.png          # Dimensionality reduction
├── 📈 clustering_visualization.png     # Cluster analysis
├── 📈 regression_comparison.png        # Model comparison
├── 📈 classification_comparison.png    # Classification results
└── 📈 confusion_matrices.png           # Confusion matrices
```

## ⚙️ Configuration

Tất cả parameters được định nghĩa trong [`Config`](config/config.py):

```python
@dataclass
class Config:
    # Paths
    DATA_FILE: str = "city_day.csv"
    
    # Model parameters
    RANDOM_STATE: int = 42
    TEST_SIZES: List[float] = [0.2, 0.3]
    
    # PCA parameters  
    PCA_COMPONENTS: int = 4
    
    # t-SNE parameters
    TSNE_COMPONENTS: int = 2
    TSNE_PERPLEXITY: int = 30
    TSNE_LEARNING_RATE: int = 200
    
    # Plotting
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    SAVE_PLOTS: bool = True
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

## 📞 Contact

- **Email**: doquocan_t67@hus.edu.vn
- **Repository**: [https://github.com/quocandev/Air-Quality-Index-Forecast](https://github.com/quocandev/Air-Quality-Index-Forecast)
- **Issues**: [https://github.com/quocandev/Air-Quality-Index-Forecast/issues](https://github.com/quocandev/Air-Quality-Index-Forecast/issues)

---

⭐ **Star this repo if you find it useful!**