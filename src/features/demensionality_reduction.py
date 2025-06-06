"""
Module giảm chiều dữ liệu với PCA và t-SNE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from config.config import Config

logger = logging.getLogger(__name__)

class DimensionalityReducer:
    """Lớp thực hiện giảm chiều dữ liệu"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne = None
    
    def reduce_dimensions(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Thực hiện giảm chiều dữ liệu bằng PCA và t-SNE
        
        Args:
            df: DataFrame đã tiền xử lý
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Dữ liệu đã giảm chiều (PCA, t-SNE)
        """
        logger.info("Bắt đầu giảm chiều dữ liệu")
        
        # Chuẩn bị dữ liệu
        X = df[self.config.POLLUTANT_COLS].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        # Thực hiện PCA
        X_pca = self._perform_pca(X_scaled)
        
        # Thực hiện t-SNE
        X_tsne = self._perform_tsne(X_scaled)
        
        # Vẽ biểu đồ so sánh
        self._plot_comparison(X_pca, X_tsne, df)
        
        logger.info("Hoàn thành giảm chiều dữ liệu")
        
        return X_pca, X_tsne
    
    def _perform_pca(self, X_scaled: np.ndarray) -> np.ndarray:
        """Thực hiện PCA"""
        logger.info("Thực hiện PCA")
        
        # PCA với tất cả components
        self.pca = PCA()
        X_pca_full = self.pca.fit_transform(X_scaled)
        
        # Phân tích phương sai giải thích
        self._analyze_pca_variance()
        
        # Vẽ biểu đồ phương sai
        self._plot_pca_variance()
        
        return X_pca_full
    
    def _perform_tsne(self, X_scaled: np.ndarray) -> np.ndarray:
        """Thực hiện t-SNE"""
        logger.info("Thực hiện t-SNE")
        
        self.tsne = TSNE(
            n_components=self.config.TSNE_COMPONENTS,
            perplexity=self.config.TSNE_PERPLEXITY,
            learning_rate=self.config.TSNE_LEARNING_RATE,
            n_iter=self.config.TSNE_N_ITER,
            random_state=self.config.RANDOM_STATE
        )
        
        X_tsne = self.tsne.fit_transform(X_scaled)
        
        return X_tsne
    
    def _analyze_pca_variance(self) -> None:
        """Phân tích phương sai giải thích của PCA"""
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        logger.info("Phương sai giải thích của PCA:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            logger.info(f"PC{i+1}: {var:.4f} ({cum_var:.4f} tích lũy)")
        
        # Tìm số components cần thiết cho 95% phương sai
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        logger.info(f"Số components cần thiết cho 95% phương sai: {n_components_95}")
    
    def _plot_pca_variance(self) -> None:
        """Vẽ biểu đồ phương sai giải thích"""
        if not self.config.SAVE_PLOTS:
            return
        
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.figure(figsize=self.config.FIGURE_SIZE)
        plt.bar(range(1, len(explained_variance)+1), explained_variance, 
                alpha=0.7, label='Phương sai riêng')
        plt.step(range(1, len(explained_variance)+1), cumulative_variance, 
                where='mid', label='Phương sai tích lũy')
        plt.axhline(y=0.95, color='r', linestyle='--', label='Ngưỡng 95%')
        plt.xlabel('Thành phần chính')
        plt.ylabel('Tỉ lệ phương sai giải thích')
        plt.title('Phương sai giải thích của các thành phần chính')
        plt.legend()
        plt.grid(True)
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('pca_variance.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def _plot_comparison(self, X_pca: np.ndarray, X_tsne: np.ndarray, 
                        df: pd.DataFrame) -> None:
        """Vẽ biểu đồ so sánh PCA và t-SNE"""
        if not self.config.SAVE_PLOTS:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Biểu đồ PCA (2 components đầu tiên)
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=df[self.config.TARGET_COL], 
                              cmap='viridis', alpha=0.6)
        ax1.set_title('PCA - 2 thành phần đầu tiên')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        plt.colorbar(scatter1, ax=ax1, label='AQI')
        
        # Biểu đồ t-SNE
        scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=df[self.config.TARGET_COL], 
                              cmap='viridis', alpha=0.6)
        ax2.set_title('t-SNE')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=ax2, label='AQI')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('pca_tsne_comparison.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def get_pca_components(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        Lấy số lượng components cụ thể từ PCA
        
        Args:
            n_components: Số lượng components cần lấy
            
        Returns:
            np.ndarray: Dữ liệu PCA với số components được chỉ định
        """
        if self.pca is None:
            raise ValueError("PCA chưa được thực hiện")
        
        if n_components is None:
            n_components = self.config.PCA_COMPONENTS
        
        return self.pca.transform(self.scaler.transform(X))[:, :n_components]