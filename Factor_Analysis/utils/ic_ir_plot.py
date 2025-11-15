"""
IC/IR 计算及折线图绘制模块
计算每个时间截面的 IC（因子与标签的截面Spearman秩相关系数），以及 IR（滑动窗口内 IC 的均值 / 标准差）。
支持多线程/异步计算以加速处理。
"""
import logging
import sys
import asyncio
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr
from utils.memory_monitor import time_perf_decorator

# 导入项目配置
CURRENT_DIR = Path(__file__).parent.parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import MAX_WORKERS, ASYNC_SEMAPHORE, USE_RANK_BASED_IC
except ImportError:
    MAX_WORKERS = 8
    ASYNC_SEMAPHORE = 10
    USE_RANK_BASED_IC = False

logger = logging.getLogger(__name__)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """计算 Spearman 秩相关系数，返回单个浮点值。样本数不足时返回 NaN。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:  # 最小样本数要求
        return np.nan
    
    # 检查数据是否有足够的独特值
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        return np.nan
        
    try:
        corr, _ = spearmanr(x, y)
        # 检查结果是否有效
        if not np.isfinite(corr):
            return np.nan
        return float(corr)
    except Exception:
        return np.nan


@time_perf_decorator()
def compute_ic_series(df: pd.DataFrame, factor_col: str, label_col: str, time_col: str = 'timestamp',
                      use_parallel: bool = True, max_workers: int = None) -> pd.Series:
    """
    计算按时间截面的 IC 序列（横截面Spearman秩相关系数）。

    要求：df 至少包含 [time_col, factor_col, label_col] 三列，并且多个 symbol 的样本在同一时间截面上。
    """
    if time_col not in df.columns:
        logger.info("数据中不存在时间列，无法计算 IC/IR。仅返回空序列。")
        return pd.Series(dtype=float)

    # 使用配置文件中的并行设置
    if max_workers is None:
        max_workers = MAX_WORKERS

    # 按时间分组
    grouped = df.groupby(time_col, sort=True)
    times = list(grouped.groups.keys())

    if not use_parallel or len(times) < 100:
        ic_values = []
        if USE_RANK_BASED_IC:
            for t, g in grouped:
                x = g[factor_col].rank(pct=True).values
                y = g[label_col].rank(pct=True).values
                if x.size < 5 or y.size < 5:
                    ic_values.append(np.nan)
                else:
                    corr = np.corrcoef(x, y)[0, 1]
                    ic_values.append(float(corr) if np.isfinite(corr) else np.nan)
        else:
            for t, g in grouped:
                x = g[factor_col].values
                y = g[label_col].values
                ic_values.append(_spearman_corr(x, y))
        ic = pd.Series(ic_values, index=list(grouped.groups.keys()), name='IC')
        ic.sort_index(inplace=True)
        return ic

    # 并行计算每个时间截面的Spearman相关系数
    def _compute_for_time(time_key) -> Tuple[pd.Timestamp, float]:
        try:
            g = grouped.get_group(time_key)
            if USE_RANK_BASED_IC:
                x = g[factor_col].rank(pct=True).values
                y = g[label_col].rank(pct=True).values
                if x.size < 5 or y.size < 5:
                    return time_key, np.nan
                corr = np.corrcoef(x, y)[0, 1]
                return time_key, float(corr) if np.isfinite(corr) else np.nan
            else:
                x = g[factor_col].values
                y = g[label_col].values
                return time_key, _spearman_corr(x, y)
        except Exception as e:
            logger.warning(f"计算时间 {time_key} 的IC时出错: {e}")
            return time_key, np.nan

    ic_map = {}
    # 使用线程池并行计算，限制并发数以确保线程安全
    with ThreadPoolExecutor(max_workers=min(max_workers, len(times))) as ex:
        futures = {ex.submit(_compute_for_time, t): t for t in times}
        for fut in as_completed(futures):
            try:
                t, val = fut.result()
                ic_map[t] = val
            except Exception as e:
                logger.warning(f"获取IC计算结果时出错: {e}")

    ic = pd.Series(ic_map, name='IC')
    ic.sort_index(inplace=True)
    logger.info(f"完成IC计算，共 {len(ic)} 个时间截面，有效值 {ic.notna().sum()} 个")
    return ic


def compute_ir_series(ic: pd.Series, window: int) -> pd.Series:
    """计算 IR 序列：在滑动窗口内的 IC 均值 / 标准差。"""
    if ic.empty or window <= 1:
        return pd.Series(dtype=float)
    roll = ic.rolling(window=window, min_periods=max(3, window//5))
    mean = roll.mean()
    std = roll.std()
    ir = mean / std
    ir.name = f'IR(window={window})'
    return ir


def _downsample_series(series: pd.Series, sampling_frequency: float = 1.0) -> pd.Series:
    """对时间序列进行下采样"""
    if sampling_frequency >= 1.0 or len(series) == 0:
        return series
    
    # 计算采样间隔
    n_total = len(series)
    n_sample = max(1, int(n_total * sampling_frequency))
    
    if n_sample >= n_total:
        return series
    
    # 等间隔采样
    indices = np.linspace(0, n_total - 1, n_sample, dtype=int)
    return series.iloc[indices]


def plot_ic_series(ic: pd.Series, title: str = "截面 IC 随时间", sampling_frequency: float = 1.0) -> plt.Figure:
    """
    绘制IC时间序列图，包含均值线、±1标准差线和统计注记
    
    Args:
        ic: IC时间序列
        title: 图表标题
        sampling_frequency: 采样频率，控制绘图点数
    """
    if ic.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '无有效IC数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # 下采样数据用于绘图
    ic_plot = _downsample_series(ic, sampling_frequency)
    
    # 计算统计量（基于全部数据）
    ic_clean = ic.dropna()
    if len(ic_clean) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '无有效IC数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    mean_val = ic_clean.mean()
    std_val = ic_clean.std()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制IC折线（细线风格）
    ax.plot(ic_plot.index, ic_plot.values, color='tab:blue', linewidth=1.0, alpha=0.8)
    
    # 绘制统计线
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'均值 = {mean_val:.4f}')
    ax.axhline(y=mean_val + std_val, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label=f'+1σ = {mean_val + std_val:.4f}')
    ax.axhline(y=mean_val - std_val, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label=f'-1σ = {mean_val - std_val:.4f}')
    
    # 设置图表属性
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('IC', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 右上角添加统计注记
    stats_text = f'Mean = {mean_val:.4f}\nStd = {std_val:.4f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 自适应坐标轴
    ax.autoscale(tight=True)
    plt.tight_layout()
    
    return fig


def plot_ir_series(ir: pd.Series, title: str = "截面 IR 随时间", sampling_frequency: float = 1.0) -> plt.Figure:
    """
    绘制IR时间序列图，包含均值线、±1标准差线和统计注记
    
    Args:
        ir: IR时间序列
        title: 图表标题
        sampling_frequency: 采样频率，控制绘图点数
    """
    if ir.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '无有效IR数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # 下采样数据用于绘图
    ir_plot = _downsample_series(ir, sampling_frequency)
    
    # 计算统计量（基于全部数据）
    ir_clean = ir.dropna()
    if len(ir_clean) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, '无有效IR数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    mean_val = ir_clean.mean()
    std_val = ir_clean.std()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制IR折线（细线风格）
    ax.plot(ir_plot.index, ir_plot.values, color='tab:orange', linewidth=1.0, alpha=0.8)
    
    # 绘制统计线
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label=f'均值 = {mean_val:.4f}')
    ax.axhline(y=mean_val + std_val, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label=f'+1σ = {mean_val + std_val:.4f}')
    ax.axhline(y=mean_val - std_val, color='orange', linestyle='--', linewidth=1.0, alpha=0.7, label=f'-1σ = {mean_val - std_val:.4f}')
    
    # 设置图表属性
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('IR', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 右上角添加统计注记
    stats_text = f'Mean = {mean_val:.4f}\nStd = {std_val:.4f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 自适应坐标轴
    ax.autoscale(tight=True)
    plt.tight_layout()
    
    return fig
from utils.memory_monitor import time_perf_decorator
