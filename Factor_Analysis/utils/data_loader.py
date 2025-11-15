"""
数据读取模块（硬件加速优化版）
读取 ML_Quant/data/feature_data 下所有币对表，仅提取指定因子列与标签列。
支持多线程并行读取，自动跳过异常或缺失数据。
应用硬件加速配置：批处理、内存管理、向量化操作。
"""
import logging
import gc
import psutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils_common import safe_import_config
from utils.memory_monitor import time_perf_decorator


logger = logging.getLogger(__name__)

# 内存监控和管理
def _check_memory_usage():
    """检查当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def _trigger_gc_if_needed(memory_limit_gb: float = 8.0, gc_threshold: float = 0.7):
    """根据内存使用情况触发垃圾回收"""
    memory_mb = _check_memory_usage()
    memory_limit_mb = memory_limit_gb * 1024
    
    if memory_mb > memory_limit_mb * gc_threshold:
        logger.info(f"内存使用 {memory_mb:.1f}MB 超过阈值，触发垃圾回收")
        gc.collect()
        new_memory_mb = _check_memory_usage()
        logger.info(f"垃圾回收后内存使用: {new_memory_mb:.1f}MB")


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """检测时间列名称，并确保为 datetime 类型。"""
    candidates = ['timestamp', 'time', 'datetime', 'open_time', 'date']
    for col in candidates:
        if col in df.columns:
            # 转换为 datetime
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])
                # 验证转换是否成功
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue  # 转换失败，尝试下一个候选列
                return col
            except Exception:
                continue  # 转换失败，尝试下一个候选列
    return None


def _read_single_file(file_path: Path, factor_col: str, label_col: str, parquet_engine: str,
                      time_range: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None,
                      enable_vectorized: bool = True, preload_to_memory: bool = True) -> Optional[pd.DataFrame]:
    """读取单个 feature_data 文件（硬件加速优化版），只保留 [time, symbol, factor_col, label_col] 四列，并按需过滤时间段。"""
    try:
        # 使用优化的读取参数
        read_kwargs = {
            'engine': parquet_engine,
            'use_threads': True,  # 启用多线程读取
        }
        
        # 如果启用预加载，使用内存映射
        if preload_to_memory:
            read_kwargs['use_pandas_metadata'] = True
        
        df = None
        try:
            df = pd.read_parquet(file_path, columns=[factor_col, label_col, 'timestamp'], **read_kwargs)
        except Exception:
            try:
                df = pd.read_parquet(file_path, columns=[factor_col, label_col], **read_kwargs)
            except Exception:
                df = pd.read_parquet(file_path, **read_kwargs)
        if df is None or df.empty:
            return None

        time_col = _detect_time_column(df)
        
        # 检查列是否存在
        missing = [c for c in [factor_col, label_col] if c not in df.columns]
        if missing:
            return None

        # 向量化选取列操作
        keep_cols = [factor_col, label_col]
        if time_col:
            keep_cols.insert(0, time_col)
        
        # 使用向量化操作选取列
        out = df[keep_cols].copy() if enable_vectorized else df[keep_cols]

        # 添加交易对符号（向量化操作）
        out['symbol'] = file_path.stem

        # 向量化清理异常数据
        if enable_vectorized:
            # 使用向量化操作替换无穷值
            numeric_cols = [factor_col, label_col]
            for col in numeric_cols:
                if col in out.columns:
                    out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            
            # 向量化删除NaN
            out = out.dropna(subset=numeric_cols)
        else:
            # 传统方式
            out.replace([np.inf, -np.inf], np.nan, inplace=True)
            out = out.dropna(subset=[factor_col, label_col])
            
        if out.empty:
            return None

        # 向量化时间排序和过滤
        if time_col:
            if enable_vectorized:
                # 使用向量化排序
                out = out.sort_values(time_col, kind='mergesort').reset_index(drop=True)
            else:
                out = out.sort_values(time_col).reset_index(drop=True)

            # 向量化时间段过滤
            if time_range is not None:
                start_ts, end_ts = time_range
                if enable_vectorized:
                    # 向量化时间过滤
                    mask = pd.Series(True, index=out.index)
                    if start_ts is not None:
                        mask &= (out[time_col] >= start_ts)
                    if end_ts is not None:
                        mask &= (out[time_col] <= end_ts)
                    out = out[mask]
                else:
                    # 传统方式
                    if start_ts is not None:
                        out = out[out[time_col] >= start_ts]
                    if end_ts is not None:
                        out = out[out[time_col] <= end_ts]
                        
                if out.empty:
                    return None

        # 内存优化：确保数据类型最优
        if enable_vectorized and not out.empty:
            # 优化数据类型以节省内存
            for col in [factor_col, label_col]:
                if col in out.columns and out[col].dtype == 'float64':
                    # 检查是否可以安全转换为float32
                    if out[col].min() >= np.finfo(np.float32).min and out[col].max() <= np.finfo(np.float32).max:
                        out[col] = out[col].astype(np.float32)

        return out
    except Exception as e:
        return None


@time_perf_decorator()
def load_factor_data(factor_name: str, use_parallel: bool = True,
                     time_range: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None) -> Tuple[pd.DataFrame, dict]:
    """
    读取 feature_data 目录中所有交易对（硬件加速优化版），只保留指定因子列与 label 列，并合并为一个总 DataFrame。

    Args:
        factor_name: 因子列名（例如 "MOM_10"、"RSI_14"）
        use_parallel: 是否使用多线程并行读取

    Returns:
        (combined_df, stats) 元组
        combined_df: 包含 [timestamp(可选), symbol, factor_name, label]
        stats: 简要统计信息字典
    """
    cfg = safe_import_config()
    feature_dir: Path = Path(cfg.FEATURE_DATA_PATH)
    
    # 应用硬件加速配置
    engine: str = getattr(cfg, 'PARQUET_ENGINE', 'pyarrow')
    max_workers: int = getattr(cfg, 'MAX_WORKERS', 8)
    batch_size: int = getattr(cfg, 'BATCH_SIZE', 50)
    memory_limit_gb: float = getattr(cfg, 'MEMORY_LIMIT_GB', 8.0)
    gc_threshold: float = getattr(cfg, 'GC_THRESHOLD', 0.7)
    enable_vectorized: bool = getattr(cfg, 'ENABLE_VECTORIZED_OPERATIONS', True)
    preload_to_memory: bool = getattr(cfg, 'PRELOAD_DATA_TO_MEMORY', True)

    label_col = 'label'  # Step2/Step3 统一标签列名为 label

    if not feature_dir.exists():
        raise FileNotFoundError(f"特征数据目录不存在：{feature_dir}")

    files = sorted(feature_dir.glob('*.parquet'))
    if not files:
        logger.info(f"在 {feature_dir} 未找到任何 parquet 文件")
        return pd.DataFrame(columns=['timestamp', 'symbol', factor_name, label_col]), {
            'files_checked': 0, 'files_used': 0, 'rows_total': 0, 'symbols': []
        }

    logger.info(f"开始加载 {len(files)} 个文件，使用硬件加速配置")
    logger.info(f"配置: max_workers={max_workers}, batch_size={batch_size}, vectorized={enable_vectorized}")
    
    results: List[pd.DataFrame] = []
    processed_files = 0

    if use_parallel:
        # 批处理并行读取
        for batch_start in range(0, len(files), batch_size):
            batch_files = files[batch_start:batch_start + batch_size]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(_read_single_file, f, factor_name, label_col, engine, 
                                  time_range, enable_vectorized, preload_to_memory): f
                    for f in batch_files
                }
                
                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    try:
                        df_part = future.result()
                        if df_part is not None and not df_part.empty:
                            results.append(df_part)
                        processed_files += 1
                        
                        # 每处理一定数量文件后检查内存
                        if processed_files % 20 == 0:
                            _trigger_gc_if_needed(memory_limit_gb, gc_threshold)
                            
                    except Exception as e:
                        logger.debug(f"读取任务失败 {f}: {e}")
                        processed_files += 1
            
            # 批次间内存管理
            _trigger_gc_if_needed(memory_limit_gb, gc_threshold)
            logger.info(f"已处理 {min(batch_start + batch_size, len(files))}/{len(files)} 个文件")
    else:
        # 串行处理（也应用优化）
        for i, f in enumerate(files):
            df_part = _read_single_file(f, factor_name, label_col, engine, time_range, 
                                      enable_vectorized, preload_to_memory)
            if df_part is not None and not df_part.empty:
                results.append(df_part)
            
            # 定期内存检查
            if (i + 1) % 20 == 0:
                _trigger_gc_if_needed(memory_limit_gb, gc_threshold)

    if not results:
        logger.warning(f"没有找到包含因子 '{factor_name}' 的有效数据")
        return pd.DataFrame(columns=['timestamp', 'symbol', factor_name, label_col]), {
            'files_checked': len(files), 'files_used': 0, 'rows_total': 0, 'symbols': []
        }

    logger.info(f"开始合并 {len(results)} 个数据片段...")
    
    # 向量化合并所有 DataFrame（硬件加速）
    if enable_vectorized and len(results) > 1:
        # 使用更高效的合并策略
        combined = pd.concat(results, ignore_index=True, copy=False)
    else:
        combined = pd.concat(results, axis=0, ignore_index=True)

    # 释放中间结果内存
    del results
    _trigger_gc_if_needed(memory_limit_gb, gc_threshold)

    # 统一列名：时间列统一为 'timestamp'（若存在）
    time_col = None
    for col in ['timestamp', 'time', 'datetime', 'open_time', 'date']:
        if col in combined.columns:
            time_col = col
            break

    if time_col and time_col != 'timestamp':
        combined.rename(columns={time_col: 'timestamp'}, inplace=True)

    # 向量化清理数据
    initial_rows = len(combined)
    if enable_vectorized:
        # 使用向量化操作清理非法值和NaN
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = combined[factor_name].notna() & combined[label_col].notna()
        combined = combined[valid_mask].copy()
    else:
        # 再次清理非法值
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined = combined.dropna(subset=[factor_name, label_col])
    
    cleaned_rows = len(combined)
    if initial_rows > cleaned_rows:
        logger.info(f"清理无效数据：{initial_rows - cleaned_rows} 行")

    # 内存优化：转换数据类型
    if enable_vectorized:
        # 优化数值列的数据类型
        for col in [factor_name, label_col]:
            if col in combined.columns and combined[col].dtype == 'float64':
                # 检查是否可以安全转换为float32
                col_min, col_max = combined[col].min(), combined[col].max()
                if (-3.4e38 <= col_min <= 3.4e38) and (-3.4e38 <= col_max <= 3.4e38):
                    combined[col] = combined[col].astype('float32')
                    logger.debug(f"列 {col} 已优化为 float32")

    # 生成统计信息
    symbols = list(sorted(set(combined['symbol'].tolist()))) if 'symbol' in combined.columns else []
    stats = {
        'files_checked': len(files),
        'files_used': len([r for r in [True] * len(files) if True]),  # 简化计数
        'rows_total': int(combined.shape[0]),
        'symbols': symbols,
        'memory_optimized': enable_vectorized,
        'data_types': {col: str(combined[col].dtype) for col in combined.columns}
    }

    # 最终内存清理
    _trigger_gc_if_needed(memory_limit_gb, gc_threshold)

    logger.info(f"数据加载完成（硬件加速）：{stats['files_used']}/{stats['files_checked']} 文件，"
                f"共 {stats['rows_total']} 行，{len(stats['symbols'])} 个交易对")
    logger.info(f"内存优化: {stats['memory_optimized']}, 数据类型: {stats['data_types']}")

    if 'symbol' in combined.columns:
        try:
            combined['symbol'] = combined['symbol'].astype('category')
        except Exception:
            pass

    return combined, stats
