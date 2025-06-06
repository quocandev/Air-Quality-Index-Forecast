"""
Module các mô hình phân cụm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Tuple, List
from config.config import Config

logger = logging.getLogger(__name__)

class ClusteringModels:
    """Lớp thực hiện các thuật toán phân cụm"""
    
    def __init__(self, config: Config):
        self.config = config
        self.kmeans_models = {}
        self.gmm_models = {}
        self.cluster_results = {}

    def perform_clustering(self, df: pd.DataFrame, X_pca: np.ndarray, 
                          X_tsne: np.ndarray) -> Dict:
        """
        Thực hiện phân cụm trên các loại dữ liệu khác nhau
        
        Args:
            df: DataFrame gốc
            X_pca: Dữ liệu đã giảm chiều bằng PCA
            X_tsne: Dữ liệu đã giảm chiều bằng t-SNE
            
        Returns:
            Dict: Kết quả phân cụm
        """
        logger.info("Bat dau thuc hien phan cum")
        
        # Chuẩn bị dữ liệu
        X_original = df[self.config.POLLUTANT_COLS].values
        
        # 1. K-means clustering
        self._perform_kmeans(X_original, X_pca, X_tsne)
        
        # 2. GMM clustering
        self._perform_gmm(X_original, X_pca, X_tsne)
        
        # 3. Đánh giá và so sánh
        self._evaluate_clustering()
        
        # 4. Trực quan hóa
        self._visualize_clustering(X_tsne, df)
        
        logger.info("Hoan thanh phan cum")
        
        return self.cluster_results

    def _perform_kmeans(self, X_original: np.ndarray, X_pca: np.ndarray, 
                       X_tsne: np.ndarray) -> None:
        """Thực hiện K-means clustering"""
        logger.info("Thuc hien K-means clustering")
        
        datasets = {
            'original': X_original,
            'pca': X_pca[:, :10],  # Lấy 10 components đầu
            'tsne': X_tsne
        }
        
        for data_name, X in datasets.items():
            # Tìm K tối ưu
            best_k = self._find_best_k_kmeans(X, data_name)
            
            # Huấn luyện mô hình với K tối ưu
            kmeans = KMeans(n_clusters=best_k, random_state=self.config.RANDOM_STATE)
            labels = kmeans.fit_predict(X)
            
            # Lưu kết quả
            self.kmeans_models[data_name] = kmeans
            self.cluster_results[f'kmeans_{data_name}'] = {
                'labels': labels,
                'n_clusters': best_k,
                'model': kmeans
            }

    def _perform_gmm(self, X_original: np.ndarray, X_pca: np.ndarray, 
                    X_tsne: np.ndarray) -> None:
        """Thực hiện Gaussian Mixture Model clustering"""
        logger.info("Thuc hien GMM clustering")
        
        datasets = {
            'original': X_original,
            'pca': X_pca[:, :10],
            'tsne': X_tsne
        }
        
        for data_name, X in datasets.items():
            # Tìm cấu hình tối ưu cho GMM
            best_gmm = self._find_best_gmm(X, data_name)
            
            # Dự đoán nhãn
            labels = best_gmm.predict(X)
            
            # Lưu kết quả
            self.gmm_models[data_name] = best_gmm
            self.cluster_results[f'gmm_{data_name}'] = {
                'labels': labels,
                'n_clusters': best_gmm.n_components,
                'model': best_gmm
            }

    def _find_best_k_kmeans(self, X: np.ndarray, data_name: str) -> int:
        """Tìm K tối ưu cho K-means bằng Silhouette Score"""
        silhouette_scores = []
        k_range = range(self.config.K_MIN, self.config.K_MAX + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        # Vẽ biểu đồ Silhouette Score
        self._plot_silhouette_scores(k_range, silhouette_scores, data_name, 'kmeans')
        
        # Trả về K có Silhouette Score cao nhất
        best_k = k_range[np.argmax(silhouette_scores)]
        
        logger.info(f"K toi uu cho K-means ({data_name}): {best_k}")
        
        return best_k
    
    def _find_best_gmm(self, X: np.ndarray, data_name: str) -> GaussianMixture:
        """Tìm cấu hình tối ưu cho GMM bằng BIC"""
        lowest_bic = np.inf
        best_gmm = None
        bic_scores = []
        
        n_components_range = range(1, 7)
        covariance_types = ['spherical', 'tied', 'diag', 'full']
        
        for covariance_type in covariance_types:
            for n_components in n_components_range:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=self.config.RANDOM_STATE
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                bic_scores.append(bic)
                
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
        
        # Vẽ biểu đồ BIC scores
        self._plot_bic_scores(n_components_range, covariance_types, 
                             bic_scores, data_name)
        
        logger.info(f"GMM tối ưu ({data_name}): {best_gmm.n_components} components, "
                   f"{best_gmm.covariance_type} covariance")
        
        return best_gmm
    
    def _evaluate_clustering(self) -> None:
        """Đánh giá chất lượng phân cụm"""
        logger.info("Đánh giá chất lượng phân cụm")
        
        evaluation_results = []
        
        for method, result in self.cluster_results.items():
            # Tính các metric đánh giá
            # (Cần có dữ liệu X tương ứng để tính)
            pass
    
    def _plot_silhouette_scores(self, k_range: range, scores: List[float], 
                               data_name: str, method: str) -> None:
        """Vẽ biểu đồ Silhouette Score"""
        if not self.config.SAVE_PLOTS:
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, scores, marker='o')
        plt.xlabel('Số cụm K')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Score cho {method.upper()} - {data_name}')
        plt.grid(True)
        
        if self.config.SAVE_PLOTS:
            filename = f'silhouette_{method}_{data_name}.png'
            plt.savefig(self.config.get_output_path(filename), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def _plot_bic_scores(self, n_components_range: range, covariance_types: List[str],
                        bic_scores: List[float], data_name: str) -> None:
        """Vẽ biểu đồ BIC scores cho GMM"""
        if not self.config.SAVE_PLOTS:
            return
        
        plt.figure(figsize=(12, 8))
        # Implementation của biểu đồ BIC
        # (Code chi tiết tương tự như trong notebook)
        
        plt.title(f'BIC scores cho GMM - {data_name}')
        if self.config.SAVE_PLOTS:
            filename = f'bic_gmm_{data_name}.png'
            plt.savefig(self.config.get_output_path(filename), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def _visualize_clustering(self, X_tsne: np.ndarray, df: pd.DataFrame) -> None:
        """Trực quan hóa kết quả phân cụm"""
        if not self.config.SAVE_PLOTS:
            return
        
        # Vẽ kết quả phân cụm trên không gian t-SNE
        methods = ['kmeans_tsne', 'gmm_tsne']
        
        fig, axes = plt.subplots(1, len(methods), figsize=(15, 6))
        
        for i, method in enumerate(methods):
            if method in self.cluster_results:
                labels = self.cluster_results[method]['labels']
                
                scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                        c=labels, cmap='viridis', alpha=0.6)
                axes[i].set_title(f'{method.upper()}')
                axes[i].set_xlabel('t-SNE 1')
                axes[i].set_ylabel('t-SNE 2')
                plt.colorbar(scatter, ax=axes[i])
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('clustering_visualization.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()