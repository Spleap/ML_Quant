"""
Factor_Analysis 公共辅助函数
- 时间戳生成
- 目录创建
- 中文字体配置
- 配置读取与安全导入
- 分箱/统计通用工具
"""
import os
import sys
import math
import logging
from pathlib import Path
from typing import Tuple, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def project_root() -> Path:
    """获取项目根目录（ML_Quant）"""
    return Path(__file__).resolve().parents[2]


def get_timestamp() -> str:
    """返回形如 YYYYMMDD_HHMMSS 的时间戳字符串"""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(dir_path: Path) -> None:
    """确保目录存在"""
    dir_path.mkdir(parents=True, exist_ok=True)


def configure_chinese_font() -> None:
    """配置 Matplotlib 中文字体，确保标题、坐标轴、图例等中文可读。
    优先选择 Windows 常见字体，其次使用 SimHei 作为备选。
    """
    try:
        # 优先使用微软雅黑或苹方（Windows/Mac 常见）
        preferred_fonts = [
            "Microsoft YaHei",  # Windows 常见
            "SimHei",            # 黑体（常见中文字体）
            "Arial Unicode MS",  # 覆盖范围广
        ]

        # 检查系统已安装字体
        available = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
        for font in preferred_fonts:
            if font in available:
                plt.rcParams['font.sans-serif'] = [font]
                break
        else:
            # 如果都不可用，退回到默认 sans-serif，但仍尝试中文显示
            plt.rcParams['font.sans-serif'] = ['sans-serif']

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # 一些美化参数
        plt.rcParams['figure.figsize'] = (8, 5)
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['savefig.dpi'] = 120
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

        logger.info("中文字体配置完成")
    except Exception as e:
        logger.info(f"中文字体配置失败：{e}")


def safe_import_config():
    """安全导入 ML_Quant/config.py，并返回模块对象。
    通过将项目根目录加入 sys.path 来保证可导入。
    """
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    try:
        cfg = importlib.import_module('config')
        return cfg
    except Exception as e:
        raise ImportError(f"无法导入全局配置 config.py：{e}")


def adaptive_bins(x: np.ndarray, method: str = 'auto', max_bins: int = 300, min_bins: int = 30) -> Tuple[np.ndarray, int]:
    """智能自适应分箱算法，根据数据分布特征选择最优分箱策略。
    
    Args:
        x: 数据一维数组（已去除 NaN）
        method: 分箱方法 ('auto', 'fd', 'scott', 'sturges', 'sqrt', 'doane', 'rice')
        max_bins: 最大分箱数量上限
        min_bins: 最小分箱数量下限
        
    Returns:
        (bin_edges, n_bins)
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    
    if n == 0:
        return np.array([-0.5, 0.5]), 1
    
    if n == 1:
        val = x[0]
        return np.array([val - 0.5, val + 0.5]), 1
    
    # 计算数据分布特征
    data_range = x.max() - x.min()
    if data_range == 0:
        val = x[0]
        return np.array([val - 0.5, val + 0.5]), 1
    
    # 计算各种分箱数量
    bins_dict = {}
    
    # Freedman-Diaconis 规则
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        fd_width = 2 * iqr / (n ** (1/3))
        bins_dict['fd'] = max(1, int(math.ceil(data_range / fd_width)))
    else:
        bins_dict['fd'] = int(math.sqrt(n))
    
    # Scott's 规则
    std = np.std(x)
    if std > 0:
        scott_width = 3.5 * std / (n ** (1/3))
        bins_dict['scott'] = max(1, int(math.ceil(data_range / scott_width)))
    else:
        bins_dict['scott'] = int(math.sqrt(n))
    
    # Sturges' 规则
    bins_dict['sturges'] = max(1, int(math.ceil(math.log2(n) + 1)))
    
    # Square root 规则
    bins_dict['sqrt'] = max(1, int(math.ceil(math.sqrt(n))))
    
    # Doane's 规则（改进的Sturges）
    if n >= 3:
        skewness = _calculate_skewness(x)
        sigma_g1 = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        bins_dict['doane'] = max(1, int(math.ceil(1 + math.log2(n) + math.log2(1 + abs(skewness) / sigma_g1))))
    else:
        bins_dict['doane'] = bins_dict['sturges']
    
    # Rice 规则
    bins_dict['rice'] = max(1, int(math.ceil(2 * (n ** (1/3)))))
    
    # 自动选择最优方法
    if method == 'auto':
        # 根据数据特征选择最适合的方法
        skewness = _calculate_skewness(x)
        kurtosis = _calculate_kurtosis(x)
        
        # 数据量较小时，使用保守的分箱数
        if n < 1000:
            candidate_bins = [bins_dict['sturges'], bins_dict['sqrt']]
        # 数据量中等时，平衡细致度和稳定性
        elif n < 10000:
            candidate_bins = [bins_dict['fd'], bins_dict['scott'], bins_dict['doane']]
        # 数据量大时，可以使用更多分箱
        else:
            candidate_bins = [bins_dict['fd'], bins_dict['scott'], bins_dict['rice']]
        
        # 根据分布特征调整
        if abs(skewness) > 1.0:  # 偏斜分布
            candidate_bins.append(bins_dict['doane'])
        if abs(kurtosis) > 3.0:  # 尖峰或平坦分布
            candidate_bins.append(bins_dict['rice'])
        
        # 选择中位数作为最终分箱数
        n_bins = int(np.median(candidate_bins))
    else:
        n_bins = bins_dict.get(method, bins_dict['fd'])
    
    # 应用边界约束
    n_bins = max(min_bins, min(max_bins, n_bins))
    
    # 生成分箱边界
    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    
    return bin_edges, n_bins


def _calculate_skewness(x: np.ndarray) -> float:
    """计算偏度"""
    n = len(x)
    if n < 3:
        return 0.0
    
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        return 0.0
    
    skew = np.mean(((x - mean) / std) ** 3)
    return skew


def _calculate_kurtosis(x: np.ndarray) -> float:
    """计算峰度"""
    n = len(x)
    if n < 4:
        return 3.0
    
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        return 3.0
    
    kurt = np.mean(((x - mean) / std) ** 4)
    return kurt


def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 200, min_bins: int = 50) -> Tuple[np.ndarray, int]:
    """使用 Freedman–Diaconis 规则计算直方图分箱（保持向后兼容）。

    Args:
        x: 数据一维数组（已去除 NaN）
        max_bins: 最大分箱数量上限，避免过度密集
        min_bins: 最小分箱数量下限，保证足够密集

    Returns:
        (bin_edges, n_bins)
    """
    return adaptive_bins(x, method='fd', max_bins=max_bins, min_bins=min_bins)


def percentile_clip(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """返回按照分位裁剪后的下界与上界，用于热力图等避免极端值影响显示。

    Args:
        x: 数据数组
        low: 下界分位数（百分比）
        high: 上界分位数（百分比）
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -1.0, 1.0
    lo = np.percentile(x, low)
    hi = np.percentile(x, high)
    if lo == hi:
        # 退化情况下扩大范围
        eps = 1e-6
        return lo - eps, hi + eps
    return float(lo), float(hi)


# 已强制不显示图像，因此不再提供顺序显示函数

def clear_dir(dir_path: Path) -> None:
    """清空目录下的所有文件与子目录（保留目录本身）。
    仅在确有需要时调用，避免误删重要文件。
    """
    try:
        if not dir_path.exists():
            return
        for p in dir_path.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    # 递归删除子目录
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)
            except Exception:
                # 静默跳过个别无法删除的文件
                pass
    except Exception:
        # 保持静默，不输出警告
        pass