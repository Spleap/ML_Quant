"""
因子统计模块
- 因子值分布直方图（实数分箱，分箱密集）
- 因子值对应标签的平均值（按同一分箱计算）
可选并行加速。
"""
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
from pathlib import Path
import sys

from .utils_common import adaptive_bins, freedman_diaconis_bins, percentile_clip
from utils.memory_monitor import time_perf_decorator


logger = logging.getLogger(__name__)

def safe_import_config():
    """安全导入配置模块"""
    try:
        # 尝试从上级目录导入 config
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        import config
        return config
    except ImportError:
        logger.warning("无法导入 config 模块，使用默认配置")
        # 返回一个具有默认值的对象
        class DefaultConfig:
            MAX_WORKERS = 8
            ENABLE_VECTORIZED_OPERATIONS = True
            MEMORY_LIMIT_GB = 8.0
            GC_THRESHOLD = 0.7
            CHUNK_SIZE = 5000
        return DefaultConfig()

def _check_memory_usage() -> float:
    """检查当前内存使用率"""
    return psutil.virtual_memory().percent / 100.0

def _trigger_gc_if_needed(memory_limit_gb: float = 8.0, threshold: float = 0.7):
    """在内存使用超过阈值时触发垃圾回收"""
    memory_usage = _check_memory_usage()
    if memory_usage > threshold:
        gc.collect()
        logger.debug(f"内存使用率 {memory_usage:.1%} 超过阈值，已执行垃圾回收")


def _chunk_indices(n: int, chunks: int) -> Tuple[int, ...]:
    step = max(1, n // chunks)
    indices = tuple(range(0, n, step))
    if indices[-1] != n:
        indices = indices + (n,)
    return indices


@time_perf_decorator()
def compute_factor_hist(df: pd.DataFrame, factor_col: str, use_parallel: bool = True, max_workers: int = None,
                        remove_outliers: bool = False, binning_method: str = 'auto') -> Dict:
    """
    计算因子值的分布直方图（硬件加速优化版），支持自适应分箱和并行计算。

    Args:
        df: 数据框
        factor_col: 因子列名
        use_parallel: 是否使用并行计算
        max_workers: 最大工作线程数
        remove_outliers: 是否移除异常值
        binning_method: 分箱方法 ('auto', 'adaptive', 'freedman_diaconis', 'fixed')

    Returns:
        包含分箱信息和统计数据的字典
    """
    # 应用硬件加速配置
    cfg = safe_import_config()
    if max_workers is None:
        max_workers = getattr(cfg, 'MAX_WORKERS', 8)
    
    enable_vectorized = getattr(cfg, 'ENABLE_VECTORIZED_OPERATIONS', True)
    memory_limit_gb = getattr(cfg, 'MEMORY_LIMIT_GB', 8.0)
    gc_threshold = getattr(cfg, 'GC_THRESHOLD', 0.7)
    chunk_size = getattr(cfg, 'CHUNK_SIZE', 5000)
    
    # 从DataFrame提取因子数据
    x = df[factor_col].values
    x = x[np.isfinite(x)]  # 预清理无效值
    
    if len(x) == 0:
        return {
            'bin_edges': np.array([]),
            'bin_centers': np.array([]),
            'counts': np.array([]),
            'n_bins': 0,
            'method_used': binning_method,
            'data_stats': {'original': {}, 'processed': {}, 'outliers_removed': remove_outliers}
        }
    
    # 向量化记录原始数据统计
    if enable_vectorized:
        # 使用向量化操作计算统计量
        original_stats = {
            'count': x.size,
            'mean': np.mean(x),
            'std': np.std(x),
            'min': np.min(x),
            'max': np.max(x)
        }
    else:
        original_stats = {
            'count': len(x),
            'mean': np.mean(x),
            'std': np.std(x),
            'min': np.min(x),
            'max': np.max(x)
        }
    
    # 向量化移除异常值
    if remove_outliers:
        if enable_vectorized:
            # 使用向量化操作进行异常值处理
            q01, q99 = np.percentile(x, [1, 99])
            mask = (x >= q01) & (x <= q99)
            x = x[mask]
        else:
            x = percentile_clip(x, lower=0.01, upper=0.99)
    
    # 向量化移除 NaN 和 Inf
    if enable_vectorized:
        finite_mask = np.isfinite(x)
        x = x[finite_mask]
    else:
        x = x[np.isfinite(x)]
    
    if len(x) == 0:
        return {
            'bin_edges': np.array([]),
            'bin_centers': np.array([]),
            'counts': np.array([]),
            'n_bins': 0,
            'method_used': binning_method,
            'data_stats': {'original': original_stats, 'processed': {}, 'outliers_removed': remove_outliers}
        }
    
    # 选择分箱方法（适配原有接口）
    if binning_method in ['auto', 'adaptive']:
        # 使用自适应分箱算法
        bin_edges, n_bins = adaptive_bins(x, method='auto', max_bins=300, min_bins=30)
    elif binning_method == 'freedman_diaconis':
        bin_edges = freedman_diaconis_bins(x)
    else:  # fixed 或其他
        if enable_vectorized:
            # 向量化计算分箱边界
            x_min, x_max = np.min(x), np.max(x)
            bin_edges = np.linspace(x_min, x_max, 50)
        else:
            bin_edges = np.linspace(np.min(x), np.max(x), 50)
    
    # 智能并行计算直方图
    parallel_threshold = chunk_size * 40  # 动态阈值
    if use_parallel and x.size > parallel_threshold and max_workers > 1:
        logger.info(f"使用并行计算直方图，数据量: {x.size:,}, 工作线程: {max_workers}")
        
        # 优化的并行分块策略
        chunks = min(max_workers, max(2, x.size // chunk_size))
        idx = _chunk_indices(x.size, chunks)
        counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
        
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i in range(len(idx) - 1):
                x_part = x[idx[i]: idx[i+1]]
                futures.append(ex.submit(np.histogram, x_part, bins=bin_edges))
            
            for fut in as_completed(futures):
                c_part, _ = fut.result()
                counts += c_part.astype(np.int64)
                
                # 定期内存检查
                _trigger_gc_if_needed(memory_limit_gb, gc_threshold)
    else:
        # 单线程向量化计算
        counts, _ = np.histogram(x, bins=bin_edges)

    # 向量化计算分箱中心点
    if enable_vectorized:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 向量化计算处理后数据统计
    if enable_vectorized:
        processed_stats = {
            'count': x.size,
            'mean': np.mean(x) if x.size > 0 else 0,
            'std': np.std(x) if x.size > 0 else 0,
            'min': np.min(x) if x.size > 0 else 0,
            'max': np.max(x) if x.size > 0 else 0
        }
    else:
        processed_stats = {
            'count': len(x),
            'mean': np.mean(x) if len(x) > 0 else 0,
            'std': np.std(x) if len(x) > 0 else 0,
            'min': np.min(x) if len(x) > 0 else 0,
            'max': np.max(x) if len(x) > 0 else 0
        }
    
    # 最终内存清理
    _trigger_gc_if_needed(memory_limit_gb, gc_threshold)
    
    return {
        'bin_edges': bin_edges,
        'bin_centers': centers,
        'counts': counts,
        'n_bins': len(bin_edges) - 1,
        'method_used': binning_method,
        'data_stats': {
            'original': original_stats,
            'processed': processed_stats,
            'outliers_removed': remove_outliers
        },
        'hardware_accelerated': enable_vectorized,
        'parallel_used': use_parallel and x.size > parallel_threshold
    }


@time_perf_decorator()
def compute_label_mean_by_bin(df: pd.DataFrame, factor_col: str, label_col: str, bin_edges: np.ndarray,
                               use_parallel: bool = False, max_workers: int = None) -> Dict:
    """
    计算每个因子分箱中标签的平均值（硬件加速优化版）。

    Returns:
        {
          'bin_edges': ndarray,
          'bin_centers': ndarray,
          'label_mean': ndarray,
          'counts': ndarray
        }
    """
    # 应用硬件加速配置
    cfg = safe_import_config()
    if max_workers is None:
        max_workers = getattr(cfg, 'MAX_WORKERS', 8)
    
    enable_vectorized = getattr(cfg, 'ENABLE_VECTORIZED_OPERATIONS', True)
    memory_limit_gb = getattr(cfg, 'MEMORY_LIMIT_GB', 8.0)
    gc_threshold = getattr(cfg, 'GC_THRESHOLD', 0.7)
    chunk_size = getattr(cfg, 'CHUNK_SIZE', 5000)
    
    # 向量化提取数据
    if enable_vectorized:
        factor_values = df[factor_col].values
        label_values = df[label_col].values
        
        # 向量化清理无效数据
        valid_mask = np.isfinite(factor_values) & np.isfinite(label_values)
        factor_values = factor_values[valid_mask]
        label_values = label_values[valid_mask]
    else:
        factor_values = df[factor_col].values
        label_values = df[label_col].values
        
        # 清理无效数据
        valid_indices = np.isfinite(factor_values) & np.isfinite(label_values)
        factor_values = factor_values[valid_indices]
        label_values = label_values[valid_indices]

    if len(factor_values) == 0:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return {
            'bin_edges': bin_edges,
            'bin_centers': centers,
            'label_mean': np.full(len(centers), np.nan),
            'counts': np.zeros(len(centers), dtype=int)
        }

    # 向量化分箱操作
    if enable_vectorized:
        # 使用numpy的digitize进行更高效的分箱
        bin_indices = np.digitize(factor_values, bin_edges) - 1
        # 确保边界值正确分配
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    else:
        # 使用 pandas.cut 划分分箱
        bins = pd.cut(factor_values, bins=bin_edges, include_lowest=True)
        tmp = pd.DataFrame({'bin': bins, 'label': label_values})

    # 智能并行处理
    parallel_threshold = chunk_size * 40
    if use_parallel and len(factor_values) > parallel_threshold and max_workers > 1:
        logger.info(f"使用并行计算分箱统计，数据量: {len(factor_values):,}")
        
        if enable_vectorized:
            n_bins = len(bin_edges) - 1
            counts = np.zeros(n_bins, dtype=np.int64)
            label_sums = np.zeros(n_bins, dtype=np.float64)
            chunk_idx = _chunk_indices(len(factor_values), max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(len(chunk_idx) - 1):
                    s, e = chunk_idx[i], chunk_idx[i + 1]
                    futures.append(executor.submit(_compute_bin_stats_chunk, bin_indices[s:e], label_values[s:e], n_bins))
                for future in as_completed(futures):
                    cs, cc = future.result()
                    label_sums += cs
                    counts += cc
                    _trigger_gc_if_needed(memory_limit_gb, gc_threshold)
            label_mean = np.divide(label_sums, counts, out=np.full_like(label_sums, np.nan), where=counts > 0)
        else:
            # 退化为单线程pandas处理
            logger.debug("数据量较大，但使用pandas groupby处理")
            grouped = tmp.groupby('bin', observed=True)
            label_mean_series = grouped['label'].mean()
            counts_series = grouped['label'].count().astype(int)
    else:
        # 单线程处理
        if enable_vectorized:
            n_bins = len(bin_edges) - 1
            counts = np.bincount(bin_indices, minlength=n_bins)
            label_sums = np.bincount(bin_indices, weights=label_values, minlength=n_bins)
            label_mean = np.divide(label_sums, counts, out=np.full_like(label_sums, np.nan), where=counts > 0)
        else:
            # pandas分组聚合
            grouped = tmp.groupby('bin', observed=True)
            label_mean_series = grouped['label'].mean()
            counts_series = grouped['label'].count().astype(int)

    # 向量化计算分箱中心点
    if enable_vectorized:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    n_bins = len(centers)
    
    if enable_vectorized:
        # 已经是numpy数组格式
        means_arr = label_mean
        counts_arr = counts
    else:
        # 处理pandas结果，对齐到所有分箱
        all_intervals = pd.cut([], bins=bin_edges, include_lowest=True).categories
        label_mean_aligned = label_mean_series.reindex(all_intervals, fill_value=np.nan)
        counts_aligned = counts_series.reindex(all_intervals, fill_value=0)
        
        means_arr = label_mean_aligned.values.astype(float)
        counts_arr = counts_aligned.values.astype(int)
    
    # 验证结果长度
    if len(means_arr) != n_bins:
        logger.warning(f"对齐后的结果长度({len(means_arr)})与预期分箱数量({n_bins})不匹配")
        if len(means_arr) > n_bins:
            means_arr = means_arr[:n_bins]
            counts_arr = counts_arr[:n_bins]
        else:
            means_arr = np.pad(means_arr, (0, n_bins - len(means_arr)), constant_values=np.nan)
            counts_arr = np.pad(counts_arr, (0, n_bins - len(counts_arr)), constant_values=0)

    # 最终内存清理
    _trigger_gc_if_needed(memory_limit_gb, gc_threshold)

    return {
        'bin_edges': bin_edges,
        'bin_centers': centers,
        'label_mean': means_arr,
        'counts': counts_arr,
        'hardware_accelerated': enable_vectorized,
        'parallel_used': use_parallel and len(factor_values) > parallel_threshold
    }

def _compute_bin_stats_chunk(bin_indices: np.ndarray, labels: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.bincount(bin_indices, weights=labels, minlength=n_bins).astype(np.float64)
    counts = np.bincount(bin_indices, minlength=n_bins).astype(np.int64)
    return sums, counts
