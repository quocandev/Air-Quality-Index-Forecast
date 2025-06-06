"""
Module load dữ liệu từ các nguồn khác nhau
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from config.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Lớp để load dữ liệu từ file CSV"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load dữ liệu từ file CSV
        
        Args:
            file_path: Đường dẫn đến file dữ liệu (tùy chọn)
            
        Returns:
            pd.DataFrame: Dữ liệu đã được load
        """
        if file_path is None:
            file_path = self.config.get_data_path()
        
        try:
            logger.info(f"Đang load dữ liệu từ: {file_path}")
            df = pd.read_csv(file_path)
            
            # Kiểm tra dữ liệu cơ bản
            logger.info(f"Kích thước dữ liệu: {df.shape}")
            logger.info(f"Các cột: {list(df.columns)}")
            logger.info(f"Số lượng giá trị thiếu: {df.isnull().sum().sum()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi load dữ liệu: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Lưu dữ liệu ra file CSV
        
        Args:
            df: DataFrame cần lưu
            file_path: Đường dẫn file output
        """
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Đã lưu dữ liệu vào: {file_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Lấy thông tin cơ bản về dữ liệu
        
        Args:
            df: DataFrame cần phân tích
            
        Returns:
            dict: Thông tin về dữ liệu
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return info