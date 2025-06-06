"""
Module trực quan hóa dữ liệu cho phân tích khám phá
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from config.config import Config

logger = logging.getLogger(__name__)

class EDAPlotter:
    """Lớp tạo các biểu đồ phân tích khám phá dữ liệu"""
    
    def __init__(self, config: Config):
        self.config = config
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Thiết lập font tiếng Việt với fallback
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Thêm cấu hình font an toàn hơn
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # Danh sách font ưu tiên hỗ trợ Unicode
        preferred_fonts = [
            'DejaVu Sans', 
            'Liberation Sans', 
            'Noto Sans', 
            'Arial', 
            'Helvetica',
            'sans-serif'
        ]
        
        # Chọn font đầu tiên có sẵn
        for font in preferred_fonts:
            if font in available_fonts or font == 'sans-serif':
                plt.rcParams['font.family'] = [font]
                break
    
    def create_all_plots(self, df: pd.DataFrame) -> None:
        """Tạo tất cả các biểu đồ EDA"""
        logger.info("Tạo tất cả biểu đồ EDA")
        
        self.plot_aqi_distribution(df)
        self.plot_pollutant_distributions(df)
        self.plot_correlation_matrix(df)
        self.plot_city_comparison(df)
        self.plot_temporal_analysis(df)
        self.plot_aqi_bucket_analysis(df)
        self.plot_missing_values(df)
        self.plot_outlier_analysis(df)
        
        logger.info("Hoàn thành tạo biểu đồ EDA")
    
    def plot_aqi_distribution(self, df: pd.DataFrame) -> None:
        """Vẽ phân phối AQI"""
        logger.info("Vẽ phân phối AQI")
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.FIGURE_SIZE)
        
        # Histogram
        axes[0,0].hist(df['AQI'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(df['AQI'].mean(), color='red', linestyle='--', 
                         label=f'Trung bình: {df["AQI"].mean():.2f}')
        axes[0,0].axvline(df['AQI'].median(), color='green', linestyle='--', 
                         label=f'Trung vị: {df["AQI"].median():.2f}')
        axes[0,0].set_title('Phân phối AQI')
        axes[0,0].set_xlabel('AQI')
        axes[0,0].set_ylabel('Tần suất')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot
        sns.boxplot(y=df['AQI'], ax=axes[0,1])
        axes[0,1].set_title('Box Plot AQI')
        
        # QQ plot
        from scipy import stats
        stats.probplot(df['AQI'].dropna(), dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot AQI')
        
        # Density plot
        df['AQI'].dropna().plot(kind='density', ax=axes[1,1])
        axes[1,1].set_title('Mật độ phân phối AQI')
        axes[1,1].set_xlabel('AQI')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('aqi_distribution.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_pollutant_distributions(self, df: pd.DataFrame) -> None:
        """Vẽ phân phối các chất ô nhiễm"""
        logger.info("Vẽ phân phối các chất ô nhiễm")
        
        n_cols = 4
        n_rows = (len(self.config.POLLUTANT_COLS) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(self.config.POLLUTANT_COLS):
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Phân phối {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Tần suất')
                    axes[i].grid(True, alpha=0.3)
        
        # Ẩn các subplot trống
        for i in range(len(self.config.POLLUTANT_COLS), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('pollutant_distributions.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """Vẽ ma trận tương quan"""
        logger.info("Vẽ ma trận tương quan")
        
        # Chọn các cột số
        numeric_cols = [col for col in self.config.NUMERIC_COLS if col in df.columns]
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Tạo mask cho nửa trên của ma trận
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Vẽ heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Ma trận tương quan giữa các chất ô nhiễm và AQI')
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('correlation_matrix.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_city_comparison(self, df: pd.DataFrame) -> None:
        """So sánh AQI giữa các thành phố"""
        logger.info("So sánh AQI giữa các thành phố")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # AQI trung bình theo thành phố
        city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
        city_aqi.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('AQI trung bình theo thành phố')
        axes[0,0].set_ylabel('AQI trung bình')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot AQI theo thành phố
        sns.boxplot(data=df, x='City', y='AQI', ax=axes[0,1])
        axes[0,1].set_title('Phân phối AQI theo thành phố')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Số lượng quan sát theo thành phố
        city_counts = df['City'].value_counts()
        city_counts.plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
        axes[1,0].set_title('Phân phối dữ liệu theo thành phố')
        axes[1,0].set_ylabel('')
        
        # Xu hướng AQI theo tháng cho từng thành phố
        for city in df['City'].unique()[:5]:  # Top 5 thành phố
            city_data = df[df['City'] == city]
            if 'Month' in city_data.columns:
                monthly_avg = city_data.groupby('Month')['AQI'].mean()
                axes[1,1].plot(monthly_avg.index, monthly_avg.values, 
                              marker='o', label=city, linewidth=2)
        
        axes[1,1].set_title('Xu hướng AQI theo tháng')
        axes[1,1].set_xlabel('Tháng')
        axes[1,1].set_ylabel('AQI trung bình')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('city_comparison.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_analysis(self, df: pd.DataFrame) -> None:
        """Phân tích xu hướng theo thời gian"""
        logger.info("Phân tích xu hướng theo thời gian")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # AQI theo năm
        if 'Year' in df.columns:
            yearly_aqi = df.groupby('Year')['AQI'].mean()
            yearly_aqi.plot(kind='line', marker='o', ax=axes[0,0], linewidth=2)
            axes[0,0].set_title('Xu hướng AQI theo năm')
            axes[0,0].set_ylabel('AQI trung bình')
            axes[0,0].grid(True, alpha=0.3)
        
        # AQI theo tháng
        if 'Month' in df.columns:
            monthly_aqi = df.groupby('Month')['AQI'].mean()
            monthly_aqi.plot(kind='bar', ax=axes[0,1], color='orange')
            axes[0,1].set_title('AQI trung bình theo tháng')
            axes[0,1].set_ylabel('AQI trung bình')
            axes[0,1].set_xlabel('Tháng')
            axes[0,1].tick_params(axis='x', rotation=0)
            axes[0,1].grid(True, alpha=0.3)
        
        # Heatmap AQI theo tháng và năm
        if 'Year' in df.columns and 'Month' in df.columns:
            pivot_data = df.pivot_table(values='AQI', index='Month', columns='Year', aggfunc='mean')
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,0])
            axes[1,0].set_title('Heatmap AQI theo tháng và năm')
        
        # Phân phối AQI theo mùa
        if 'Month' in df.columns:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Đông'
                elif month in [3, 4, 5]:
                    return 'Xuân'
                elif month in [6, 7, 8]:
                    return 'Hè'
                else:
                    return 'Thu'
            
            df_temp = df.copy()
            df_temp['Season'] = df_temp['Month'].apply(get_season)
            sns.boxplot(data=df_temp, x='Season', y='AQI', ax=axes[1,1])
            axes[1,1].set_title('Phân phối AQI theo mùa')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('temporal_analysis.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_aqi_bucket_analysis(self, df: pd.DataFrame) -> None:
        """Phân tích AQI_Bucket"""
        logger.info("Phân tích AQI_Bucket")
        
        if 'AQI_Bucket' not in df.columns:
            logger.warning("Không tìm thấy cột AQI_Bucket")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Phân phối AQI_Bucket
        bucket_counts = df['AQI_Bucket'].value_counts()
        bucket_counts.plot(kind='bar', ax=axes[0,0], color='lightcoral')
        axes[0,0].set_title('Phân phối mức độ chất lượng không khí')
        axes[0,0].set_ylabel('Số lượng')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Pie chart
        bucket_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
        axes[0,1].set_title('Tỷ lệ mức độ chất lượng không khí')
        axes[0,1].set_ylabel('')
        
        # AQI_Bucket theo thành phố
        bucket_city = pd.crosstab(df['City'], df['AQI_Bucket'], normalize='index') * 100
        bucket_city.plot(kind='bar', stacked=True, ax=axes[1,0])
        axes[1,0].set_title('Tỷ lệ mức độ chất lượng không khí theo thành phố (%)')
        axes[1,0].set_ylabel('Tỷ lệ (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='AQI_Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # AQI_Bucket theo tháng
        if 'Month' in df.columns:
            bucket_month = pd.crosstab(df['Month'], df['AQI_Bucket'], normalize='index') * 100
            bucket_month.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Tỷ lệ mức độ chất lượng không khí theo tháng (%)')
            axes[1,1].set_xlabel('Tháng')
            axes[1,1].set_ylabel('Tỷ lệ (%)')
            axes[1,1].tick_params(axis='x', rotation=0)
            axes[1,1].legend(title='AQI_Bucket')
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('aqi_bucket_analysis.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_missing_values(self, df: pd.DataFrame) -> None:
        """Phân tích giá trị thiếu"""
        logger.info("Phân tích giá trị thiếu")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Cột': missing_data.index,
            'Số lượng thiếu': missing_data.values,
            'Tỷ lệ thiếu (%)': missing_percent.values
        })
        missing_df = missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Số lượng thiếu', ascending=False)
        
        if len(missing_df) == 0:
            logger.info("Không có giá trị thiếu trong dữ liệu")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot số lượng thiếu
        axes[0].bar(range(len(missing_df)), missing_df['Số lượng thiếu'])
        axes[0].set_title('Số lượng giá trị thiếu theo cột')
        axes[0].set_xlabel('Cột')
        axes[0].set_ylabel('Số lượng thiếu')
        axes[0].set_xticks(range(len(missing_df)))
        axes[0].set_xticklabels(missing_df['Cột'], rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Bar plot tỷ lệ thiếu
        axes[1].bar(range(len(missing_df)), missing_df['Tỷ lệ thiếu (%)'], color='orange')
        axes[1].set_title('Tỷ lệ giá trị thiếu theo cột (%)')
        axes[1].set_xlabel('Cột')
        axes[1].set_ylabel('Tỷ lệ thiếu (%)')
        axes[1].set_xticks(range(len(missing_df)))
        axes[1].set_xticklabels(missing_df['Cột'], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('missing_values_analysis.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def plot_outlier_analysis(self, df: pd.DataFrame) -> None:
        """Phân tích giá trị ngoại lai"""
        logger.info("Phân tích giá trị ngoại lai")
        
        numeric_cols = [col for col in self.config.POLLUTANT_COLS + ['AQI'] if col in df.columns]
        
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            data = df[col].dropna()
            if len(data) > 0:
                sns.boxplot(y=data, ax=axes[i])
                axes[i].set_title(f'Box Plot - {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Ẩn các subplot trống
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if self.config.SAVE_PLOTS:
            plt.savefig(self.config.get_output_path('outlier_analysis.png'), 
                       dpi=self.config.DPI, bbox_inches='tight')
        plt.show()
    
    def create_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo bảng thống kê tóm tắt"""
        logger.info("Tạo bảng thống kê tóm tắt")
        
        numeric_cols = [col for col in self.config.NUMERIC_COLS if col in df.columns]
        summary_stats = df[numeric_cols].describe()
        
        # Thêm các thống kê bổ sung
        additional_stats = pd.DataFrame(index=['missing_count', 'missing_percent'])
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            additional_stats[col] = [missing_count, missing_percent]
        
        # Kết hợp các thống kê
        summary_stats = pd.concat([summary_stats, additional_stats])
        
        # Lưu kết quả
        summary_stats.to_csv(self.config.get_output_path('summary_statistics.csv'))
        
        return summary_stats