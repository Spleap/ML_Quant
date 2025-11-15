"""
热力图绘制模块
生成因子值 vs 标签 的热力图，采样密集，密集区域自适应，确保热点清晰。
"""
import gc
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制使用非GUI后端
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import psutil

from .utils_common import freedman_diaconis_bins, percentile_clip


logger = logging.getLogger(__name__)


def safe_import_config():
    """安全导入配置，如果失败则使用默认值"""
    try:
        # 尝试从项目根目录导入config
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        import config
        return {
            'ENABLE_VECTORIZED_OPERATIONS': getattr(config, 'ENABLE_VECTORIZED_OPERATIONS', True),
            'MEMORY_LIMIT_GB': getattr(config, 'MEMORY_LIMIT_GB', 8),
            'GC_THRESHOLD': getattr(config, 'GC_THRESHOLD', 0.8),
            'MAX_WORKERS': getattr(config, 'MAX_WORKERS', None),
            'CHUNK_SIZE': getattr(config, 'CHUNK_SIZE', 10000)
        }
    except ImportError:
        logger.warning("无法导入config模块，使用默认配置")
        return {
            'ENABLE_VECTORIZED_OPERATIONS': True,
            'MEMORY_LIMIT_GB': 8,
            'GC_THRESHOLD': 0.8,
            'MAX_WORKERS': None,
            'CHUNK_SIZE': 10000
        }


def _check_memory_usage():
    """检查当前内存使用情况"""
    try:
        memory_info = psutil.virtual_memory()
        memory_usage_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        logger.debug(f"当前内存使用: {memory_usage_gb:.2f}GB ({memory_percent:.1f}%)")
        return memory_usage_gb, memory_percent
    except Exception as e:
        logger.warning(f"无法获取内存信息: {e}")
        return 0, 0


def _trigger_gc_if_needed(threshold: float = 0.8):
    """如果内存使用超过阈值则触发垃圾回收"""
    try:
        memory_usage_gb, memory_percent = _check_memory_usage()
        if memory_percent / 100 > threshold:
            logger.info(f"内存使用率 {memory_percent:.1f}% 超过阈值 {threshold*100:.1f}%，触发垃圾回收")
            gc.collect()
            # 再次检查内存
            new_usage_gb, new_percent = _check_memory_usage()
            logger.info(f"垃圾回收后内存使用: {new_usage_gb:.2f}GB ({new_percent:.1f}%)")
    except Exception as e:
        logger.warning(f"垃圾回收检查失败: {e}")


def _optimize_heatmap_params(memory_limit_gb: float) -> dict:
    """根据内存限制优化热力图参数"""
    if memory_limit_gb <= 4:
        return {
            'figsize': (8, 6),
            'dpi': 100,
            'max_bins': 100,
            'use_rasterized': True
        }
    elif memory_limit_gb <= 8:
        return {
            'figsize': (10, 8),
            'dpi': 150,
            'max_bins': 150,
            'use_rasterized': False
        }
    else:
        return {
            'figsize': (12, 10),
            'dpi': 200,
            'max_bins': 200,
            'use_rasterized': False
        }


def create_factor_label_heatmap(factor: np.ndarray, label: np.ndarray,
                                title: str = "因子值 vs 标签 热力图") -> plt.Figure:
    """
    根据因子和标签的数值，生成二维直方图热力图。

    - 横轴为因子值，纵轴为标签值
    - 使用分位数裁剪范围，避免极端值影响显示
    - 分箱数量依据 Freedman–Diaconis 规则，保证密集程度
    - 对于密集区域使用对数归一化以便热点更清晰
    """
    # 初始内存检查
    _check_memory_usage()
    
    # 导入配置设置
    config = safe_import_config()
    enable_vectorized = config.get('ENABLE_VECTORIZED_OPERATIONS', True)
    memory_limit_gb = config.get('MEMORY_LIMIT_GB', 8)
    gc_threshold = config.get('GC_THRESHOLD', 0.8)
    
    # 优化热力图参数
    heatmap_params = _optimize_heatmap_params(memory_limit_gb)
    
    # 向量化数据清理
    if enable_vectorized:
        x = np.asarray(factor, dtype=np.float32)
        y = np.asarray(label, dtype=np.float32)
        # 向量化有效性检查
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
    else:
        x = np.asarray(factor)
        y = np.asarray(label)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

    if x.size == 0 or y.size == 0:
        fig, ax = plt.subplots(figsize=heatmap_params['figsize'], dpi=heatmap_params['dpi'])
        ax.text(0.5, 0.5, "数据为空，无法生成热力图", ha='center', va='center')
        ax.set_axis_off()
        return fig

    # 向量化裁剪显示范围
    if enable_vectorized:
        x_lo, x_hi = np.percentile(x, [1.0, 99.0])
        y_lo, y_hi = np.percentile(y, [1.0, 99.0])
    else:
        x_lo, x_hi = percentile_clip(x, 1.0, 99.0)
        y_lo, y_hi = percentile_clip(y, 1.0, 99.0)

    # 智能计算分箱数量
    max_bins = heatmap_params['max_bins']
    if enable_vectorized:
        # 向量化分箱计算
        x_edges, x_n = freedman_diaconis_bins(x)
        y_edges, y_n = freedman_diaconis_bins(y)
        
        # 根据内存限制调整分箱数量
        x_n = min(max(x_n, max_bins//2), max_bins)
        y_n = min(max(y_n, max_bins//2), max_bins)
        x_edges = np.linspace(x_lo, x_hi, x_n + 1, dtype=np.float32)
        y_edges = np.linspace(y_lo, y_hi, y_n + 1, dtype=np.float32)
    else:
        x_edges, x_n = freedman_diaconis_bins(x)
        y_edges, y_n = freedman_diaconis_bins(y)
        x_n = min(max(x_n, 100), max_bins)
        y_n = min(max(y_n, 100), max_bins)
        x_edges = np.linspace(x_lo, x_hi, x_n + 1)
        y_edges = np.linspace(y_lo, y_hi, y_n + 1)

    # 向量化二维直方图计算
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=heatmap_params['figsize'], dpi=heatmap_params['dpi'])
    # 设置更优雅的背景色
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')  # 淡灰色背景，更加柔和
    
    # 向量化处理0值，避免LogNorm错误
    if enable_vectorized:
        H_safe = np.where(H > 0, H, 1e-10)  # 将0值替换为极小正数
        
        # 向量化计算颜色范围
        nonzero_mask = H > 0
        if np.any(nonzero_mask):
            nonzero_values = H[nonzero_mask]
            vmin = max(1, np.percentile(nonzero_values, 5))
            vmax = np.percentile(nonzero_values, 95)
        else:
            vmin, vmax = 1, 1
    else:
        H_safe = np.where(H > 0, H, 1e-10)
        nonzero_values = H[H > 0]
        if len(nonzero_values) > 0:
            vmin = max(1, np.percentile(nonzero_values, 5))
            vmax = np.percentile(nonzero_values, 95)
        else:
            vmin, vmax = 1, 1
    
    # 使用黄绿蓝配色方案（添加光栅化选项）
    mesh = ax.pcolormesh(xedges, yedges, H_safe.T, 
                        cmap='viridis',  # 使用viridis颜色映射，黄绿蓝配色，清爽美观
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        shading='auto',
                        rasterized=heatmap_params.get('use_rasterized', False))
    
    # 优化颜色条设置
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('样本数量 (对数尺度)', fontsize=12, color='#2c3e50')
    cb.ax.tick_params(labelsize=10, colors='#2c3e50')
    cb.outline.set_edgecolor('#dee2e6')
    cb.outline.set_linewidth(1)

    # 优化标题和标签设置
    ax.set_title(title, fontsize=14, color='#2c3e50', pad=20)
    ax.set_xlabel('因子值', fontsize=12, color='#2c3e50')
    ax.set_ylabel('标签值', fontsize=12, color='#2c3e50')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    
    # 优化坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#495057')
    
    # 优化网格样式
    ax.grid(True, alpha=0.2, color='#dee2e6', linewidth=0.5)
    ax.set_axisbelow(True)  # 将网格置于图形下方
    
    # 优化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#dee2e6')
        spine.set_linewidth(1)
    
    # 调整布局，确保标签完整显示
    plt.tight_layout()
    
    # 内存管理和清理
    _trigger_gc_if_needed(gc_threshold)

    return fig