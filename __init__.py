"""
Dự án phân tích dữ liệu chất lượng không khí và xây dựng mô hình dự đoán
Nhóm 5: Trần Kiều Hạnh, Đỗ Quốc An, Phạm Thị Duyên
"""

__version__ = "1.0.0"
__author__ = "Nhóm 5 - Trần Kiều Hạnh, Đỗ Quốc An, Phạm Thị Duyên"
__email__ = "group5@example.com"
__description__ = "Air Quality Analysis and Prediction Model"

# Import các module chính
from src import models, features, visualization
from data import data_loader, preprocessing
from config import config

__all__ = [
    'models',
    'features', 
    'visualization',
    'data_loader',
    'preprocessing',
    'config'
]