"""
Cấu hình font cho matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging

logger = logging.getLogger(__name__)

def setup_matplotlib_fonts():
    """Thiết lập font an toàn cho matplotlib"""
    try:
        # Lấy danh sách font có sẵn
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # Danh sách font ưu tiên hỗ trợ Unicode/Tiếng Việt
        preferred_fonts = [
            'DejaVu Sans',
            'Liberation Sans', 
            'Noto Sans',
            'Ubuntu',
            'Arial',
            'Helvetica',
            'Calibri',
            'Segoe UI',
            'sans-serif'
        ]
        
        # Tìm font đầu tiên có sẵn
        selected_font = 'sans-serif'  # fallback mặc định
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        # Cấu hình matplotlib
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        logger.info(f"Đã thiết lập font: {selected_font}")
        return selected_font
        
    except Exception as e:
        logger.warning(f"Lỗi thiết lập font: {e}")
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return 'sans-serif'

def get_available_fonts():
    """Trả về danh sách font có sẵn"""
    return sorted([f.name for f in fm.fontManager.ttflist])

def check_font_support(font_name: str) -> bool:
    """Kiểm tra font có được hỗ trợ không"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name in available_fonts