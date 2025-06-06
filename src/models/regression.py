"""
Module các mô hình hồi quy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import gc  # Thêm garbage collection
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple
from config.config import Config
import psutil
import os

logger = logging.getLogger(__name__)

class RegressionModels:
    """Lớp thực hiện các mô hình hồi quy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}

    def train_models(self, df: pd.DataFrame, X_pca: np.ndarray, 
                    X_tsne: np.ndarray, cluster_results: Dict) -> Dict:
        """
        Huấn luyện các mô hình hồi quy
        """
        logger.info("Bắt đầu huấn luyện mô hình hồi quy")
        
        # Chuẩn bị dữ liệu
        datasets = self._prepare_datasets(df, X_pca, X_tsne)
        
        # 1. Random Forest Regressor
        self._train_random_forest(datasets, cluster_results)
        
        # Giải phóng RAM
        gc.collect()
        
        # 2. MLP Regressor
        self._train_mlp(datasets, cluster_results)
        
        # Giải phóng RAM
        gc.collect()
        
        # 3. Phân tích phần dư (skip nếu không cần thiết)
        # self._analyze_residuals(datasets)
        
        # 4. So sánh mô hình
        self._compare_models()
        
        logger.info("Hoàn thành huấn luyện mô hình hồi quy")
        
        return self.results
    
    def _prepare_datasets(self, df: pd.DataFrame, X_pca: np.ndarray, 
                         X_tsne: np.ndarray) -> Dict:
        """Chuẩn bị các dataset khác nhau"""
        target = df[self.config.TARGET_COL]
        
        datasets = {
            'original': {
                'X': df[self.config.POLLUTANT_COLS + self.config.TIME_COLS].values,
                'y': target,
                'feature_names': self.config.POLLUTANT_COLS + self.config.TIME_COLS
            },
            'pca': {
                'X': X_pca,
                'y': target,
                'feature_names': [f'PC{i+1}' for i in range(X_pca.shape[1])]
            },
            'tsne': {
                'X': X_tsne,
                'y': target,
                'feature_names': ['TSNE1', 'TSNE2']
            }
        }
        
        return datasets
    
    def log_memory_usage(self):  # ✅ Thêm self
        """Log memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"RAM usage: {memory_info.rss / 1024 / 1024:.1f} MB")

    def _train_random_forest(self, datasets: Dict, cluster_results: Dict) -> None:
        """Huấn luyện Random Forest Regressor"""
        logger.info("Huấn luyện Random Forest Regressor")
        self.log_memory_usage()

        for data_name, data in datasets.items():
            X, y = data['X'], data['y']
            
            for test_size in self.config.TEST_SIZES:
                for n_estimators in self.config.N_ESTIMATORS_LIST:
                    # Chia dữ liệu
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, 
                        random_state=self.config.RANDOM_STATE
                    )
                    
                    # Huấn luyện mô hình
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=self.config.RANDOM_STATE,
                        n_jobs=-1
                    )
                    rf.fit(X_train, y_train)
                    
                    # Đánh giá
                    metrics = self._evaluate_regression_model(rf, X_test, y_test)
                    
                    # Lưu kết quả
                    key = f"rf_{data_name}_{test_size}_{n_estimators}"
                    self.models[key] = rf
                    self.results[key] = {
                        'model_type': 'RandomForest',
                        'data_type': data_name,
                        'test_size': test_size,
                        'n_estimators': n_estimators,
                        'metrics': metrics
                    }
    
    def _train_mlp(self, datasets: Dict, cluster_results: Dict) -> None:
        """Huấn luyện MLP Regressor"""
        logger.info("Huấn luyện MLP Regressor")
        self.log_memory_usage()
        
        for data_name, data in datasets.items():
            X, y = data['X'], data['y']
            
            for test_size in self.config.TEST_SIZES:
                for hidden_size in self.config.HIDDEN_SIZES:
                    # Chia dữ liệu
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, 
                        random_state=self.config.RANDOM_STATE
                    )
                    
                    # Huấn luyện mô hình
                    mlp = MLPRegressor(
                        hidden_layer_sizes=(hidden_size,),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        random_state=self.config.RANDOM_STATE
                    )
                    mlp.fit(X_train, y_train)
                    
                    # Đánh giá
                    metrics = self._evaluate_regression_model(mlp, X_test, y_test)
                    
                    # Lưu kết quả
                    key = f"mlp_{data_name}_{test_size}_{hidden_size}"
                    self.models[key] = mlp
                    self.results[key] = {
                        'model_type': 'MLP',
                        'data_type': data_name,
                        'test_size': test_size,
                        'hidden_size': hidden_size,
                        'metrics': metrics
                    }
    
    def _evaluate_regression_model(self, model, X_test: np.ndarray, 
                                  y_test: np.ndarray) -> Dict:
        """Đánh giá mô hình hồi quy"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics
    
    def _analyze_residuals(self, datasets: Dict) -> None:
        """Phân tích phần dư"""
        logger.info("Phân tích phần dư")
        
        # Implementation phân tích phần dư
        # Tính toán và trực quan hóa phần dư
        pass
    
    def _compare_models(self) -> None:
        """So sánh các mô hình"""
        logger.info("So sánh các mô hình")
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame([
            {
                'Model': result['model_type'],
                'Data': result['data_type'],
                'Test_Size': result['test_size'],
                'MSE': result['metrics']['mse'],
                'RMSE': result['metrics']['rmse'],
                'R2': result['metrics']['r2'],
                'MAE': result['metrics']['mae']
            }
            for result in self.results.values()
        ])
        
        # Vẽ biểu đồ so sánh
        self._plot_model_comparison(results_df)
        
        # Lưu kết quả
        results_df.to_csv(self.config.get_output_path('regression_results.csv'), 
                         index=False)
    
    def _plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """Vẽ biểu đồ so sánh mô hình"""
        if not self.config.SAVE_PLOTS:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R2 Score
        sns.barplot(data=results_df, x='Model', y='R2', hue='Data', ax=axes[0,0])
        axes[0,0].set_title('R² Score Comparison')
        
        # RMSE
        sns.barplot(data=results_df, x='Model', y='RMSE', hue='Data', ax=axes[0,1])
        axes[0,1].set_title('RMSE Comparison')
        
        # MAE
        sns.barplot(data=results_df, x='Model', y='MAE', hue='Data', ax=axes[1,0])
        axes[1,0].set_title('MAE Comparison')
        
        # MSE
        sns.barplot(data=results_df, x='Model', y='MSE', hue='Data', ax=axes[1,1])
        axes[1,1].set_title('MSE Comparison')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('regression_comparison.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()