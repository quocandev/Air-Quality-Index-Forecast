"""
File cấu hình chính cho dự án
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Config:
    """Lớp cấu hình chính cho dự án"""
    
    # Đường dẫn
    ROOT_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    OUTPUT_DIR: Path = ROOT_DIR / "output"
    DOC_DIR: Path = ROOT_DIR / "doc"
    
    # File dữ liệu
    DATA_FILE: str = "city_day.csv"
    
    # Cột dữ liệu
    NUMERIC_COLS: List[str] = None
    POLLUTANT_COLS: List[str] = None
    TIME_COLS: List[str] = None
    TARGET_COL: str = "AQI"
    
    # Tham số mô hình
    RANDOM_STATE: int = 42
    TEST_SIZES: List[float] = None
    N_ESTIMATORS_LIST: List[int] = None
    HIDDEN_SIZES: List[int] = None
    
    # Tham số PCA
    PCA_COMPONENTS: int = 4
    
    # Tham số t-SNE
    TSNE_COMPONENTS: int = 2
    TSNE_PERPLEXITY: int = 30
    TSNE_LEARNING_RATE: int = 200
    TSNE_N_ITER: int = 3000
    
    # Tham số phân cụm
    K_MIN: int = 2
    K_MAX: int = 5
    
    # Cấu hình vẽ biểu đồ
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    SAVE_PLOTS: bool = True
    
    def __post_init__(self):
        """Khởi tạo các thuộc tính sau khi tạo object"""
        if self.NUMERIC_COLS is None:
            self.NUMERIC_COLS = [
                'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
                'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'
            ]
        
        if self.POLLUTANT_COLS is None:
            self.POLLUTANT_COLS = [
                'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
            ]
        
        if self.TIME_COLS is None:
            self.TIME_COLS = ['Year', 'Month', 'Day']
        
        if self.TEST_SIZES is None:
            self.TEST_SIZES = [0.2, 0.3, 0.4]
        
        if self.N_ESTIMATORS_LIST is None:
            self.N_ESTIMATORS_LIST = [50, 100]
        
        if self.HIDDEN_SIZES is None:
            self.HIDDEN_SIZES = [50, 75]
        
        # Tạo thư mục nếu chưa tồn tại
        self.OUTPUT_DIR.mkdir(exist_ok=True)

    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.RAW_DATA_DIR.mkdir(exist_ok=True)
        self.DOC_DIR.mkdir(exist_ok=True)
    
    def get_data_path(self) -> Path:
        """Trả về đường dẫn đến file dữ liệu chính"""
        return self.RAW_DATA_DIR / self.DATA_FILE
    
    def get_output_path(self, filename: str) -> Path:
        """Trả về đường dẫn đến file output"""
        return self.OUTPUT_DIR / filename