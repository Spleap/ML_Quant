"""
增强的图表绘制模块
提供更精美、细致的因子分析图表展示功能
硬件加速优化版：支持内存管理、向量化操作、批量处理。
"""
import logging
import gc
import psutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from .utils_common import safe_import_config

logger = logging.getLogger(__name__)


def _check_memory_usage() -> float:
    """检查当前内存使用率"""
    try:
        return psutil.virtual_memory().percent / 100.0
    except Exception:
        return 0.0


def _trigger_gc_if_needed(memory_threshold: float = 0.7):
    """如果内存使用超过阈值，触发垃圾回收"""
    if _check_memory_usage() > memory_threshold:
        gc.collect()
        logger.debug(f"内存使用超过{memory_threshold*100:.1f}%，已触发垃圾回收")


def _optimize_figure_params():
    """获取优化的图像参数"""
    cfg = safe_import_config()
    
    # 根据内存限制调整图像参数
    memory_limit_gb = getattr(cfg, 'MEMORY_LIMIT_GB', 8.0)
    
    if memory_limit_gb <= 4:
        # 低内存模式
        return {
            'dpi': 100,
            'figsize_scale': 0.8,
            'max_points': 10000,
            'use_rasterized': True
        }
    elif memory_limit_gb <= 8:
        # 中等内存模式
        return {
            'dpi': 150,
            'figsize_scale': 1.0,
            'max_points': 50000,
            'use_rasterized': False
        }
    else:
        # 高内存模式
        return {
            'dpi': 200,
            'figsize_scale': 1.2,
            'max_points': 100000,
            'use_rasterized': False
        }


def create_enhanced_factor_histogram(hist_data: Dict, factor_name: str, remove_outliers: bool = False) -> plt.Figure:
    """
    创建增强的因子分布直方图（硬件加速优化版）
    
    Args:
        hist_data: 直方图数据字典
        factor_name: 因子名称
        remove_outliers: 是否去除了极值
        
    Returns:
        matplotlib Figure对象
    """
    # 应用硬件加速配置
    cfg = safe_import_config()
    gc_threshold = getattr(cfg, 'GC_THRESHOLD', 0.7)
    enable_vectorized = getattr(cfg, 'ENABLE_VECTORIZED_OPERATIONS', True)
    
    # 获取优化参数
    fig_params = _optimize_figure_params()
    figsize = (12 * fig_params['figsize_scale'], 10 * fig_params['figsize_scale'])
    
    # 检查初始内存
    initial_memory = _check_memory_usage()
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], 
                                       dpi=fig_params['dpi'])
        
        # 主直方图数据
        bin_edges = hist_data['bin_edges']
        bin_centers = hist_data['bin_centers']
        counts = hist_data['counts']
        
        # 向量化计算柱状图宽度
        if enable_vectorized:
            widths = np.diff(bin_edges)
        else:
            widths = bin_edges[1:] - bin_edges[:-1]
        
        # 使用渐变色彩（向量化）
        if enable_vectorized:
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
        else:
            colors = [plt.cm.viridis(0.2 + 0.6 * i / len(counts)) for i in range(len(counts))]
        
        # 绘制主直方图
        bars = ax1.bar(bin_centers, counts, width=widths, color=colors, 
                       edgecolor='white', linewidth=0.5, alpha=0.8,
                       rasterized=fig_params['use_rasterized'])
        
        # 向量化添加数值标签（仅对较高的柱子）
        if enable_vectorized:
            max_count = np.max(counts)
            high_bars_mask = counts > max_count * 0.1
            high_centers = bin_centers[high_bars_mask]
            high_counts = counts[high_bars_mask]
            
            for center, count in zip(high_centers, high_counts):
                ax1.text(center, count + max_count * 0.01, f'{count:,}', 
                        ha='center', va='bottom', fontsize=8, color='#2c3e50')
        else:
            max_count = np.max(counts)
            for i, (center, count, bar) in enumerate(zip(bin_centers, counts, bars)):
                if count > max_count * 0.1:  # 只标注高度超过最大值10%的柱子
                    ax1.text(center, count + max_count * 0.01, f'{count:,}', 
                            ha='center', va='bottom', fontsize=8, color='#2c3e50')
        
        # 美化主图
        title1 = f"图1：{factor_name} 因子值分布（自适应分箱）"
        if remove_outliers:
            title1 += "（已去极值）"

        ax1.set_title(title1, fontsize=14, color='#2c3e50', pad=20)
        ax1.set_xlabel('因子值', fontsize=12, color='#2c3e50')
        ax1.set_ylabel('频数', fontsize=12, color='#2c3e50')
        ax1.grid(True, alpha=0.3, color='#dee2e6', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # 设置坐标轴样式
        ax1.tick_params(axis='both', which='major', labelsize=10, colors='#495057')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#dee2e6')
            spine.set_linewidth(1)
        
        # 添加统计信息文本框
        stats = hist_data.get('data_stats', {})
        if 'processed' in stats:
            processed = stats['processed']
            info_text = (f"样本数: {processed['count']:,}\n"
                        f"均值: {processed['mean']:.4f}\n"
                        f"标准差: {processed['std']:.4f}\n"
                        f"分箱数: {hist_data['n_bins']}\n"
                        f"分箱方法: {hist_data.get('method_used', 'auto')}")
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8, edgecolor='#dee2e6'),
                    fontsize=9, color='#2c3e50')
        
        # 下方子图：累积分布
        cumulative = np.cumsum(counts)
        cumulative_pct = cumulative / cumulative[-1] * 100
        
        ax2.plot(bin_centers, cumulative_pct, color='#e74c3c', linewidth=2, marker='o', 
                 markersize=3, alpha=0.8, label='累积分布')
        ax2.fill_between(bin_centers, 0, cumulative_pct, alpha=0.3, color='#e74c3c')
        
        ax2.set_xlabel('因子值', fontsize=10, color='#2c3e50')
        ax2.set_ylabel('累积百分比 (%)', fontsize=10, color='#2c3e50')
        ax2.grid(True, alpha=0.3, color='#dee2e6', linewidth=0.5)
        ax2.set_axisbelow(True)
        ax2.tick_params(axis='both', which='major', labelsize=9, colors='#495057')
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#dee2e6')
            spine.set_linewidth(1)
        
        # 设置x轴范围一致
        ax1.set_xlim(bin_edges[0], bin_edges[-1])
        ax2.set_xlim(bin_edges[0], bin_edges[-1])
        
        plt.tight_layout()
        
        # 内存管理和清理
        _trigger_gc_if_needed(gc_threshold)
        
        return fig
    
    except Exception as e:
        logger.error(f"创建因子直方图时发生错误: {e}")
        # 创建一个简单的错误图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"生成图表时发生错误:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig


def create_enhanced_label_mean_plot(mean_data: Dict, factor_name: str, remove_outliers: bool = False) -> plt.Figure:
    """
    创建增强的因子分箱标签均值图
    
    Args:
        mean_data: 标签均值数据字典
        factor_name: 因子名称
        remove_outliers: 是否去除了极值
        
    Returns:
        matplotlib Figure对象
    """
    # 初始内存检查
    _check_memory_usage()
    
    # 导入配置设置
    config = safe_import_config()
    enable_vectorized = getattr(config, 'ENABLE_VECTORIZED_OPERATIONS', True)
    memory_limit_gb = getattr(config, 'MEMORY_LIMIT_GB', 8)
    gc_threshold = getattr(config, 'GC_THRESHOLD', 0.8)
    
    # 优化图形参数
    fig_params = _optimize_figure_params()
    
    figsize = (12 * fig_params['figsize_scale'], 8 * fig_params['figsize_scale'])
    fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=fig_params['dpi'])
    
    # 向量化数据提取
    if enable_vectorized:
        bin_centers = np.asarray(mean_data['bin_centers'], dtype=np.float32)
        label_means = np.asarray(mean_data['label_mean'], dtype=np.float32)
        counts = np.asarray(mean_data['counts'], dtype=np.int32)
        bin_edges = np.asarray(mean_data['bin_edges'], dtype=np.float32)
        
        # 向量化过滤NaN值
        valid_mask = np.isfinite(label_means)
        valid_centers = bin_centers[valid_mask]
        valid_means = label_means[valid_mask]
        valid_counts = counts[valid_mask]
    else:
        bin_centers = mean_data['bin_centers']
        label_means = mean_data['label_mean']
        counts = mean_data['counts']
        bin_edges = mean_data['bin_edges']
        
        # 过滤掉NaN值
        valid_mask = np.isfinite(label_means)
        valid_centers = bin_centers[valid_mask]
        valid_means = label_means[valid_mask]
        valid_counts = counts[valid_mask]
    
    if len(valid_centers) == 0:
        # 如果没有有效数据，显示空图
        ax1.text(0.5, 0.5, '无有效数据', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14, color='#e74c3c')
        ax1.set_title(f"图2：{factor_name} 因子分箱标签均值", fontsize=14, color='#2c3e50')
        return fig
    
    # 向量化计算柱状图宽度
    if enable_vectorized:
        widths = np.diff(bin_edges)
        valid_widths = widths[valid_mask]
        
        # 向量化颜色设置
        colors = np.where(valid_means >= 0, '#27ae60', '#e74c3c')
        
        # 向量化计算标签位置
        mean_range = np.max(valid_means) - np.min(valid_means) if len(valid_means) > 1 else 1.0
        label_offset = 0.02 * mean_range
        label_y = valid_means + np.where(valid_means >= 0, label_offset, -label_offset)
    else:
        widths = bin_edges[1:] - bin_edges[:-1]
        valid_widths = widths[valid_mask]
        colors = ['#27ae60' if mean >= 0 else '#e74c3c' for mean in valid_means]
    
    # 绘制主柱状图（使用光栅化以提高性能）
    bars = ax1.bar(valid_centers, valid_means, width=valid_widths, 
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.8,
                   rasterized=fig_params.get('use_rasterized', False))
    
    # 添加零线
    ax1.axhline(y=0, color='#34495e', linestyle='-', linewidth=1, alpha=0.7)
    
    # 智能添加数值标签（避免过多标签影响性能）
    max_labels = fig_params.get('max_points', 50)
    if len(valid_centers) <= max_labels:
        if enable_vectorized:
            # 向量化标签添加
            for i, (center, mean, count) in enumerate(zip(valid_centers, valid_means, valid_counts)):
                if count > 0:
                    ax1.text(center, label_y[i], f'{mean:.4f}', ha='center', 
                            va='bottom' if mean >= 0 else 'top', fontsize=8, color='#2c3e50')
        else:
            for center, mean, count in zip(valid_centers, valid_means, valid_counts):
                if count > 0:
                    label_y_single = mean + (0.02 if mean >= 0 else -0.02) * (np.max(valid_means) - np.min(valid_means))
                    ax1.text(center, label_y_single, f'{mean:.4f}', ha='center', 
                            va='bottom' if mean >= 0 else 'top', fontsize=8, color='#2c3e50')
    
    # 美化主图
    title2 = f"图2：{factor_name} 因子分箱标签均值（自适应分箱）"
    if remove_outliers:
        title2 += "（已去极值）"
    
    ax1.set_title(title2, fontsize=14, color='#2c3e50', pad=20)
    ax1.set_xlabel('因子值（分箱中心）', fontsize=12, color='#2c3e50')
    ax1.set_ylabel('标签平均值', fontsize=12, color='#2c3e50')
    ax1.grid(True, alpha=0.3, color='#dee2e6', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # 设置坐标轴样式
    ax1.tick_params(axis='both', which='major', labelsize=10, colors='#495057')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#dee2e6')
        spine.set_linewidth(1)
    
    # 添加统计信息
    valid_mean = np.mean(valid_means)
    valid_std = np.std(valid_means)
    monotonicity = _calculate_monotonicity(valid_centers, valid_means)
    
    info_text = (f"有效分箱数: {len(valid_centers)}\n"
                f"标签均值: {valid_mean:.4f}\n"
                f"标签标准差: {valid_std:.4f}\n"
                f"单调性: {monotonicity:.3f}")
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8, edgecolor='#dee2e6'),
            fontsize=9, color='#2c3e50')
    
    # 设置x轴范围
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    
    plt.tight_layout()
    
    # 内存管理和清理
    _trigger_gc_if_needed(gc_threshold)
    
    return fig


def _calculate_monotonicity(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算单调性指标（Spearman相关系数）
    
    Args:
        x: x值数组
        y: y值数组
        
    Returns:
        单调性指标 (-1到1之间)
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    
    try:
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(x, y)
        return correlation if np.isfinite(correlation) else 0.0
    except ImportError:
        # 如果没有scipy，使用简单的线性相关系数
        return np.corrcoef(x, y)[0, 1] if np.isfinite(np.corrcoef(x, y)[0, 1]) else 0.0


def create_enhanced_heatmap(factor: np.ndarray, label: np.ndarray, 
                          factor_name: str, title: str = None) -> plt.Figure:
    """
    创建增强的因子-标签热力图
    
    Args:
        factor: 因子值数组
        label: 标签值数组
        factor_name: 因子名称
        title: 图表标题
        
    Returns:
        matplotlib Figure对象
    """
    from .utils_common import adaptive_bins, percentile_clip
    
    # 清理数据
    x = np.asarray(factor)
    y = np.asarray(label)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0 or y.size == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "数据为空，无法生成热力图", ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='#e74c3c')
        ax.set_title(title or f"图3：{factor_name} vs 标签 热力图", fontsize=14, color='#2c3e50')
        return fig

    # 使用自适应分箱
    x_edges, x_n = adaptive_bins(x, method='auto', max_bins=100, min_bins=20)
    y_edges, y_n = adaptive_bins(y, method='auto', max_bins=100, min_bins=20)

    # 二维直方图
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    
    # 处理0值，避免LogNorm错误
    H_safe = np.where(H > 0, H, np.nan)
    
    # 使用更精美的颜色映射
    cmap = plt.cm.plasma
    cmap.set_bad(color='#f8f9fa', alpha=0.3)  # 设置NaN值的颜色
    
    # 计算更合理的颜色范围
    nonzero_values = H[H > 0]
    if len(nonzero_values) > 0:
        vmin = np.percentile(nonzero_values, 10)
        vmax = np.percentile(nonzero_values, 90)
    else:
        vmin, vmax = 1, 1
    
    # 绘制热力图
    mesh = ax.pcolormesh(xedges, yedges, H_safe.T, 
                        cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    # 优化颜色条
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label('样本密度', fontsize=12, color='#2c3e50')
    cb.ax.tick_params(labelsize=10, colors='#2c3e50')
    cb.outline.set_edgecolor('#dee2e6')
    cb.outline.set_linewidth(1)

    # 设置标题和标签
    if title is None:
        title = f"图3：{factor_name} vs 标签 热力图（自适应分箱）"
    
    ax.set_title(title, fontsize=14, color='#2c3e50', pad=20)
    ax.set_xlabel(f'{factor_name} 因子值', fontsize=12, color='#2c3e50')
    ax.set_ylabel('标签值', fontsize=12, color='#2c3e50')
    
    # 优化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#495057')
    
    # 优化网格
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 优化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#dee2e6')
        spine.set_linewidth(1)
    
    # 添加统计信息
    correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    info_text = (f"样本数: {len(x):,}\n"
                f"相关系数: {correlation:.4f}\n"
                f"X分箱数: {x_n}\n"
                f"Y分箱数: {y_n}")
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='#dee2e6'),
            fontsize=9, color='#2c3e50')
    
    plt.tight_layout()
    return fig