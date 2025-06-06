"""
File chính để chạy toàn bộ pipeline phân tích dữ liệu chất lượng không khí
"""

import os
import sys
import logging
from pathlib import Path
import locale

# Thiết lập encoding cho Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

class UTF8StreamHandler(logging.StreamHandler):
    """Custom StreamHandler với UTF-8 encoding"""
    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8')
            except (AttributeError, OSError):
                pass

def setup_logging_early():
    """Thiết lập logging sớm với UTF-8 encoding"""
    # Tạo thư mục output nếu chưa có
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Xóa handlers cũ
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Tạo formatter (không có encoding parameter)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler với UTF-8
    file_handler = logging.FileHandler(
        output_dir / 'project.log', 
        mode='a', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler với UTF-8
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Thiết lập logging sớm
setup_logging_early()
logger = logging.getLogger(__name__)

def setup_logging(config):
    """Thiết lập logging cho dự án với UTF-8 encoding"""
    # Tạo thư mục output nếu chưa có
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Xóa handlers cũ
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Tạo formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler với UTF-8
    file_handler = logging.FileHandler(
        config.OUTPUT_DIR / 'project.log', 
        mode='a', 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler với UTF-8
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def main():
    """Hàm chính chạy toàn bộ pipeline"""
    try:
        # 1. Load cấu hình trước
        from config.config import Config
        config = Config()
        logger.info("Da load cau hinh thanh cong")
        
        # 2. Thiết lập logging với UTF-8
        setup_logging(config)
        
        # 3. Thiết lập font (với error handling)
        try:
            from utils.font_config import setup_matplotlib_fonts
            setup_matplotlib_fonts()
            logger.info("Da thiet lap font thanh cong")
        except ImportError as e:
            logger.warning(f"Khong the import font_config: {e}")
            # Thiết lập font fallback
            import matplotlib.pyplot as plt
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            logger.info("Da thiet lap font fallback")
        
        logger.info("Bat dau chay pipeline phan tich du lieu chat luong khong khi")
        
        # 4. Import các module khác
        from data.data_loader import DataLoader
        from data.preprocessing import DataPreprocessor
        from src.features.demensionality_reduction import DimensionalityReducer
        from src.models.clustering import ClusteringModels
        from src.models.regression import RegressionModels
        from src.models.classification import ClassificationModels
        from src.visualization.eda_plots import EDAPlotter
        
        # 5. Load dữ liệu
        data_loader = DataLoader(config)
        df = data_loader.load_data()
        logger.info(f"Da load du lieu thanh cong: {df.shape}")
        
        # 6. Tiền xử lý dữ liệu
        preprocessor = DataPreprocessor(config)
        df_processed = preprocessor.preprocess(df)
        logger.info("Da tien xu ly du lieu thanh cong")
        
        # 7. Phân tích thống kê mô tả và trực quan hóa
        plotter = EDAPlotter(config)
        plotter.create_all_plots(df_processed)
        logger.info("Da tao cac bieu do EDA")
        
        # 8. Giảm chiều dữ liệu
        dim_reducer = DimensionalityReducer(config)
        X_pca, X_tsne = dim_reducer.reduce_dimensions(df_processed)
        logger.info("Da thuc hien giam chieu du lieu")
        
        # 9. Phân cụm
        clustering = ClusteringModels(config)
        cluster_results = clustering.perform_clustering(df_processed, X_pca, X_tsne)
        logger.info("Da thuc hien phan cum")
        
        # 10. Mô hình hồi quy
        regression = RegressionModels(config)
        regression_results = regression.train_models(df_processed, X_pca, X_tsne, cluster_results)
        logger.info("Da huan luyen mo hinh hoi quy")
        
        # 11. Mô hình phân loại
        classification = ClassificationModels(config)
        classification_results = classification.train_models(df_processed, X_pca)
        logger.info("Da huan luyen mo hinh phan loai")
        
        logger.info("Hoan thanh pipeline thanh cong!")
        
    except Exception as e:
        logger.error(f"Loi khi chay pipeline: {str(e)}")
        import traceback
        logger.error(f"Chi tiet loi:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()