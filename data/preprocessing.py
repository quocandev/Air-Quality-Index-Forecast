"""
Module tiền xử lý dữ liệu
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
from config.config import Config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Lớp tiền xử lý dữ liệu"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = MinMaxScaler()
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện toàn bộ quá trình tiền xử lý dữ liệu
        
        Args:
            df: DataFrame gốc
            
        Returns:
            pd.DataFrame: Dữ liệu đã được tiền xử lý
        """
        logger.info("Bắt đầu tiền xử lý dữ liệu")
        
        df_processed = df.copy()
        
        # 1. Chuyển đổi cột Date
        df_processed = self._process_date_column(df_processed)
        
        # 2. Xử lý giá trị thiếu
        df_processed = self._handle_missing_values(df_processed)
        
        # 3. Xử lý giá trị ngoại lai
        df_processed = self._handle_outliers(df_processed)
        
        # 4. Xử lý AQI_Bucket
        df_processed = self._process_aqi_bucket(df_processed)
        
        # 5. Tạo biến dummy
        df_processed = self._create_dummy_variables(df_processed)
        
        # 6. Chuẩn hóa dữ liệu
        df_normalized = self._normalize_data(df_processed)
        
        logger.info("Hoàn thành tiền xử lý dữ liệu")
        
        return df_normalized
    
    def _process_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý cột Date và tạo các cột thời gian"""
        logger.info("Xử lý cột Date")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý giá trị thiếu"""
        logger.info("Xử lý giá trị thiếu")
        
        # Điền giá trị thiếu bằng trung bình theo thành phố
        for col in self.config.NUMERIC_COLS:
            if col in df.columns:
                city_means = df.groupby('City')[col].transform('mean')
                df[col] = df[col].fillna(city_means)
                
                # Nếu vẫn còn NA, điền bằng trung bình toàn bộ
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý giá trị ngoại lai bằng phương pháp IQR"""
        logger.info("Xử lý giá trị ngoại lai")
        
        for col in self.config.NUMERIC_COLS:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Thay thế giá trị ngoại lai
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _process_aqi_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý cột AQI_Bucket"""
        logger.info("Xử lý cột AQI_Bucket")
        
        def map_aqi_to_bucket(aqi):
            if pd.isna(aqi):
                return None
            elif aqi <= 50:
                return 'Good'
            elif aqi <= 100:
                return 'Satisfactory'
            elif aqi <= 200:
                return 'Moderate'
            elif aqi <= 300:
                return 'Poor'
            elif aqi <= 400:
                return 'Very Poor'
            else:
                return 'Severe'
        
        # Điền giá trị thiếu cho AQI_Bucket
        df['AQI_Bucket'] = df.apply(
            lambda row: row['AQI_Bucket'] if pd.notna(row['AQI_Bucket']) 
            else map_aqi_to_bucket(row['AQI']), axis=1
        )
        
        return df
    
    def _create_dummy_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo biến dummy cho các biến categorical"""
        logger.info("Tạo biến dummy")
        
        # One-hot encoding cho AQI_Bucket
        if 'AQI_Bucket' in df.columns:
            aqi_bucket_dummies = pd.get_dummies(df['AQI_Bucket'], prefix='AQI_Bucket')
            df = pd.concat([df, aqi_bucket_dummies], axis=1)
        
        # One-hot encoding cho City
        if 'City' in df.columns:
            city_dummies = pd.get_dummies(df['City'], prefix='City')
            df = pd.concat([df, city_dummies], axis=1)
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa dữ liệu về thang đo 0-1"""
        logger.info("Chuẩn hóa dữ liệu")
        
        df_normalized = df.copy()
        
        for col in self.config.NUMERIC_COLS:
            if col in df.columns:
                df_normalized[col] = self.scaler.fit_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()
        
        return df_normalized
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Tách features và target
        
        Args:
            df: DataFrame đã tiền xử lý
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features và target
        """
        features = df[self.config.POLLUTANT_COLS + self.config.TIME_COLS].copy()
        target = df[self.config.TARGET_COL].copy()
        
        return features, target