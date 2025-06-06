"""
Module các mô hình phân loại
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import KBinsDiscretizer
from typing import Dict, List, Tuple
from config.config import Config

logger = logging.getLogger(__name__)

class ClassificationModels:
    """Lớp thực hiện các mô hình phân loại"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.results = {}
        self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', 
                                          strategy='quantile')
    
    def train_models(self, df: pd.DataFrame, X_pca: np.ndarray) -> Dict:
        """
        Huấn luyện các mô hình phân loại
        
        Args:
            df: DataFrame gốc
            X_pca: Dữ liệu PCA
            
        Returns:
            Dict: Kết quả các mô hình phân loại
        """
        logger.info("Bắt đầu huấn luyện mô hình phân loại")
        
        # 1. Chuẩn bị dữ liệu phân loại
        df_classification = self._prepare_classification_data(df)
        
        # 2. Chuẩn bị datasets
        datasets = self._prepare_datasets(df_classification, X_pca)
        
        # 3. Huấn luyện Naive Bayes
        self._train_naive_bayes(datasets)
        
        # 4. Huấn luyện Random Forest Classifier
        self._train_random_forest_classifier(datasets)
        
        # 5. So sánh mô hình
        self._compare_classification_models()
        
        logger.info("Hoàn thành huấn luyện mô hình phân loại")
        
        return self.results
    
    def _prepare_classification_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn bị dữ liệu cho bài toán phân loại"""
        logger.info("Chuẩn bị dữ liệu phân loại")
        
        df_classification = df.copy()
        
        # Chia AQI thành 4 khoảng có số lượng mẫu xấp xỉ nhau
        aqi_classes = self.discretizer.fit_transform(df[[self.config.TARGET_COL]])
        df_classification['AQI_class_numeric'] = aqi_classes.flatten()
        
        # Tạo nhãn có ý nghĩa
        class_mapping = {0: 'Thấp', 1: 'Trung bình', 2: 'Cao', 3: 'Rất cao'}
        df_classification['AQI_class'] = df_classification['AQI_class_numeric'].map(class_mapping)
        
        # Hiển thị phân phối các lớp
        class_distribution = df_classification['AQI_class'].value_counts()
        logger.info(f"Phân phối các lớp: \n{class_distribution}")
        
        return df_classification
    
    def _prepare_datasets(self, df: pd.DataFrame, X_pca: np.ndarray) -> Dict:
        """Chuẩn bị datasets cho phân loại"""
        features_original = df[self.config.POLLUTANT_COLS + self.config.TIME_COLS]
        target = df['AQI_class_numeric']
        
        # Giảm chiều PCA xuống 1/3 số chiều ban đầu
        n_components_reduced = len(self.config.POLLUTANT_COLS) // 3
        X_pca_reduced = X_pca[:, :n_components_reduced]
        
        datasets = {
            'original': {
                'X': features_original,
                'y': target,
                'feature_names': features_original.columns.tolist()
            },
            'pca': {
                'X': pd.DataFrame(X_pca_reduced, 
                                columns=[f'PC{i+1}' for i in range(n_components_reduced)]),
                'y': target,
                'feature_names': [f'PC{i+1}' for i in range(n_components_reduced)]
            }
        }
        
        return datasets
    
    def _train_naive_bayes(self, datasets: Dict) -> None:
        """Huấn luyện mô hình Naive Bayes"""
        logger.info("Huấn luyện Naive Bayes")
        
        for data_name, data in datasets.items():
            X, y = data['X'], data['y']
            
            for test_size in self.config.TEST_SIZES:
                # Chia dữ liệu
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, 
                    random_state=self.config.RANDOM_STATE,
                    stratify=y
                )
                
                # Huấn luyện mô hình
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                
                # Đánh giá
                metrics = self._evaluate_classification_model(nb, X_test, y_test)
                
                # Lưu kết quả
                key = f"nb_{data_name}_{test_size}"
                self.models[key] = nb
                self.results[key] = {
                    'model_type': 'NaiveBayes',
                    'data_type': data_name,
                    'test_size': test_size,
                    'metrics': metrics
                }
    
    def _train_random_forest_classifier(self, datasets: Dict) -> None:
        """Huấn luyện Random Forest Classifier"""
        logger.info("Huấn luyện Random Forest Classifier")
        
        for data_name, data in datasets.items():
            X, y = data['X'], data['y']
            
            for test_size in self.config.TEST_SIZES:
                for n_estimators in self.config.N_ESTIMATORS_LIST:
                    # Chia dữ liệu
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, 
                        random_state=self.config.RANDOM_STATE,
                        stratify=y
                    )
                    
                    # Huấn luyện mô hình
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=self.config.RANDOM_STATE,
                        n_jobs=-1
                    )
                    rf.fit(X_train, y_train)
                    
                    # Đánh giá
                    metrics = self._evaluate_classification_model(rf, X_test, y_test)
                    
                                        # Lưu kết quả
                    key = f"rf_clf_{data_name}_{test_size}_{n_estimators}"
                    self.models[key] = rf
                    self.results[key] = {
                        'model_type': 'RandomForestClassifier',
                        'data_type': data_name,
                        'test_size': test_size,
                        'n_estimators': n_estimators,
                        'metrics': metrics
                    }
    
    def _evaluate_classification_model(self, model, X_test: np.ndarray, 
                                     y_test: np.ndarray) -> Dict:
        """Đánh giá mô hình phân loại"""
        y_pred = model.predict(X_test)
        
        # Tính các metric cơ bản
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Ma trận nhầm lẫn
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        return metrics
    
    def _compare_classification_models(self) -> None:
        """So sánh các mô hình phân loại"""
        logger.info("So sánh các mô hình phân loại")
        
        # Tạo DataFrame kết quả
        results_list = []
        for key, result in self.results.items():
            results_list.append({
                'Model': result['model_type'],
                'Data': result['data_type'],
                'Test_Size': result['test_size'],
                'N_Estimators': result.get('n_estimators', 'N/A'),
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1_Score': result['metrics']['f1_score']
            })
        
        results_df = pd.DataFrame(results_list)
        
        # Vẽ biểu đồ so sánh
        self._plot_classification_comparison(results_df)
        
        # Vẽ confusion matrix cho mô hình tốt nhất
        self._plot_best_confusion_matrices()
        
        # Lưu kết quả
        results_df.to_csv(self.config.get_output_path('classification_results.csv'), 
                         index=False)
        
        # In kết quả tốt nhất
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        logger.info(f"Mô hình tốt nhất: {best_model['Model']} - "
                   f"Data: {best_model['Data']} - "
                   f"Accuracy: {best_model['Accuracy']:.4f}")
    
    def _plot_classification_comparison(self, results_df: pd.DataFrame) -> None:
        """Vẽ biểu đồ so sánh các mô hình phân loại"""
        if not self.config.SAVE_PLOTS:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Accuracy', hue='Data', ax=axes[0,0])
        axes[0,0].set_title('Accuracy Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        sns.barplot(data=results_df, x='Model', y='Precision', hue='Data', ax=axes[0,1])
        axes[0,1].set_title('Precision Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        sns.barplot(data=results_df, x='Model', y='Recall', hue='Data', ax=axes[1,0])
        axes[1,0].set_title('Recall Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        sns.barplot(data=results_df, x='Model', y='F1_Score', hue='Data', ax=axes[1,1])
        axes[1,1].set_title('F1-Score Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('classification_comparison.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def _plot_best_confusion_matrices(self) -> None:
        """Vẽ confusion matrix cho các mô hình tốt nhất"""
        if not self.config.SAVE_PLOTS:
            return
        
        # Tìm mô hình tốt nhất cho mỗi loại dữ liệu
        best_models = {}
        for key, result in self.results.items():
            data_type = result['data_type']
            accuracy = result['metrics']['accuracy']
            
            if data_type not in best_models or accuracy > best_models[data_type]['accuracy']:
                best_models[data_type] = {
                    'key': key,
                    'accuracy': accuracy,
                    'confusion_matrix': result['metrics']['confusion_matrix'],
                    'model_type': result['model_type']
                }
        
        # Vẽ confusion matrix
        n_models = len(best_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        class_names = ['Thấp', 'Trung bình', 'Cao', 'Rất cao']
        
        for i, (data_type, model_info) in enumerate(best_models.items()):
            cm = np.array(model_info['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, 
                       ax=axes[i])
            axes[i].set_title(f'Confusion Matrix\n{model_info["model_type"]} - {data_type}\n'
                            f'Accuracy: {model_info["accuracy"]:.4f}')
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('confusion_matrices.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self, datasets: Dict) -> None:
        """Vẽ biểu đồ tầm quan trọng của features cho Random Forest"""
        if not self.config.SAVE_PLOTS:
            return
        
        # Tìm mô hình Random Forest tốt nhất
        best_rf_models = {}
        for key, result in self.results.items():
            if result['model_type'] == 'RandomForestClassifier':
                data_type = result['data_type']
                accuracy = result['metrics']['accuracy']
                
                if data_type not in best_rf_models or accuracy > best_rf_models[data_type]['accuracy']:
                    best_rf_models[data_type] = {
                        'model': self.models[key],
                        'accuracy': accuracy,
                        'feature_names': datasets[data_type]['feature_names']
                    }
        
        # Vẽ feature importance
        n_models = len(best_rf_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (data_type, model_info) in enumerate(best_rf_models.items()):
            model = model_info['model']
            feature_names = model_info['feature_names']
            importances = model.feature_importances_
            
            # Sắp xếp theo tầm quan trọng
            indices = np.argsort(importances)[::-1]
            
            # Vẽ biểu đồ
            axes[i].bar(range(len(importances)), importances[indices])
            axes[i].set_title(f'Feature Importance - {data_type}\n'
                            f'Accuracy: {model_info["accuracy"]:.4f}')
            axes[i].set_xlabel('Features')
            axes[i].set_ylabel('Importance')
            axes[i].set_xticks(range(len(importances)))
            axes[i].set_xticklabels([feature_names[i] for i in indices], 
                                  rotation=45, ha='right')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('feature_importance_classification.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, object, Dict]:
        """
        Lấy mô hình tốt nhất theo metric được chỉ định
        
        Args:
            metric: Metric để đánh giá ('accuracy', 'precision', 'recall', 'f1_score')
            
        Returns:
            Tuple[str, object, Dict]: Key, model object, và kết quả của mô hình tốt nhất
        """
        if not self.results:
            raise ValueError("Chưa có mô hình nào được huấn luyện")
        
        best_key = None
        best_score = -1
        
        for key, result in self.results.items():
            score = result['metrics'][metric]
            if score > best_score:
                best_score = score
                best_key = key
        
        return best_key, self.models[best_key], self.results[best_key]
    
    def predict_aqi_class(self, features: np.ndarray, model_key: str = None) -> np.ndarray:
        """
        Dự đoán lớp AQI cho dữ liệu mới
        
        Args:
            features: Dữ liệu đầu vào
            model_key: Key của mô hình cần sử dụng (nếu None, sử dụng mô hình tốt nhất)
            
        Returns:
            np.ndarray: Nhãn lớp được dự đoán
        """
        if model_key is None:
            model_key, _, _ = self.get_best_model()
        
        model = self.models[model_key]
        predictions = model.predict(features)
        
        # Chuyển đổi về nhãn có ý nghĩa
        class_mapping = {0: 'Thấp', 1: 'Trung bình', 2: 'Cao', 3: 'Rất cao'}
        predicted_labels = [class_mapping[pred] for pred in predictions]
        
        return np.array(predicted_labels)
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt kết quả tất cả các mô hình
        
        Returns:
            pd.DataFrame: Bảng tóm tắt kết quả
        """
        summary_data = []
        
        for key, result in self.results.items():
            summary_data.append({
                'Model_Key': key,
                'Model_Type': result['model_type'],
                'Data_Type': result['data_type'],
                'Test_Size': result['test_size'],
                'Parameters': str(result.get('n_estimators', 'N/A')),
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1_Score': result['metrics']['f1_score']
            })
        
        return pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)