"""
Step3 特征工程辅助函数
包括数据加载、保存、并行处理等功能
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LABELED_DATA_PATH, FEATURE_DATA_PATH, MAX_WORKERS, 
    CHUNK_SIZE, PARQUET_ENGINE, PARQUET_COMPRESSION
)

logger = logging.getLogger(__name__)

def load_labeled_data_sync(symbol: str) -> Optional[pd.DataFrame]:
    """
    同步加载单个交易对的标签数据
    
    Args:
        symbol: 交易对符号
        
    Returns:
        DataFrame或None
    """
    try:
        file_path = Path(LABELED_DATA_PATH) / f"{symbol}.parquet"
        
        if not file_path.exists():
            logger.warning(f"标签数据文件不存在: {file_path}")
            return None
        
        df = pd.read_parquet(file_path, engine=PARQUET_ENGINE)
        
        # 基本验证
        if df.empty:
            logger.warning(f"{symbol}: 标签数据为空")
            return None
        
        # 确保时间戳列存在且为datetime类型
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"{symbol}: 成功加载标签数据 {len(df)} 行")
        return df
        
    except Exception as e:
        logger.error(f"{symbol}: 加载标签数据失败 - {str(e)}")
        return None

async def load_labeled_data_async(symbol: str, executor: ThreadPoolExecutor) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    异步加载单个交易对的标签数据
    
    Args:
        symbol: 交易对符号
        executor: 线程池执行器
        
    Returns:
        (symbol, DataFrame或None)
    """
    try:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(executor, load_labeled_data_sync, symbol)
        return symbol, df
        
    except Exception as e:
        logger.error(f"{symbol}: 异步加载标签数据失败 - {str(e)}")
        return symbol, None

async def load_all_labeled_data_async(symbols: List[str] = None, max_workers: int = MAX_WORKERS) -> Dict[str, pd.DataFrame]:
    """
    异步加载所有标签数据
    
    Args:
        symbols: 指定的交易对列表，None表示加载所有
        max_workers: 最大工作线程数
        
    Returns:
        {symbol: dataframe} 字典
    """
    try:
        labeled_data_dir = Path(LABELED_DATA_PATH)
        
        if not labeled_data_dir.exists():
            logger.error(f"标签数据目录不存在: {labeled_data_dir}")
            return {}
        
        # 获取所有可用的标签数据文件
        if symbols is None:
            parquet_files = list(labeled_data_dir.glob("*.parquet"))
            symbols = [f.stem for f in parquet_files]
        
        if not symbols:
            logger.warning("没有找到标签数据文件")
            return {}
        
        logger.info(f"开始异步加载 {len(symbols)} 个交易对的标签数据")
        
        # 创建线程池和异步任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [load_labeled_data_async(symbol, executor) for symbol in symbols]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        data_dict = {}
        successful_loads = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"加载任务异常: {str(result)}")
                continue
            
            symbol, df = result
            if df is not None:
                data_dict[symbol] = df
                successful_loads += 1
        
        logger.info(f"成功加载 {successful_loads}/{len(symbols)} 个交易对的标签数据")
        return data_dict
        
    except Exception as e:
        logger.error(f"异步加载所有标签数据失败: {str(e)}")
        return {}

def save_feature_data_sync(symbol: str, df: pd.DataFrame) -> bool:
    """
    同步保存单个交易对的特征数据
    
    Args:
        symbol: 交易对符号
        df: 要保存的DataFrame
        
    Returns:
        是否成功
    """
    try:
        output_dir = Path(FEATURE_DATA_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{symbol}.parquet"
        
        # 保存前的数据验证
        if df.empty:
            logger.warning(f"{symbol}: 数据为空，跳过保存")
            return False
        
        # 清理因子计算不足导致的缺失值
        # 只清理因子列的缺失值，其他列的缺失值保留
        cleaned_df = df.copy()
        
        # 识别因子列（RSI_14, MOM_10等）
        factor_columns = [col for col in cleaned_df.columns if 
                         col.startswith('RSI_') or col.startswith('MOM_')]
        
        if factor_columns:
            initial_rows = len(cleaned_df)
            
            # 删除任何因子列为NaN的行
            # RSI_14前14行、MOM_10前10行等会有NaN值
            cleaned_df = cleaned_df.dropna(subset=factor_columns)
            
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                logger.info(f"{symbol}: 清理了 {removed_rows} 行因子计算不足导致的缺失数据")
        
        # 确保清理后仍有数据
        if cleaned_df.empty:
            logger.warning(f"{symbol}: 清理后数据为空，跳过保存")
            return False
        
        # 确保时间戳列为datetime类型
        if 'timestamp' in cleaned_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        
        # 保存为Parquet格式
        cleaned_df.to_parquet(
            file_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False
        )
        
        logger.info(f"{symbol}: 成功保存特征数据到 {file_path} ({len(cleaned_df)} 行, {len(cleaned_df.columns)} 列)")
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: 保存特征数据失败 - {str(e)}")
        return False

async def save_feature_data_async(symbol: str, df: pd.DataFrame, executor: ThreadPoolExecutor) -> Tuple[str, bool]:
    """
    异步保存单个交易对的特征数据
    
    Args:
        symbol: 交易对符号
        df: 要保存的DataFrame
        executor: 线程池执行器
        
    Returns:
        (symbol, 是否成功)
    """
    try:
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(executor, save_feature_data_sync, symbol, df)
        return symbol, success
        
    except Exception as e:
        logger.error(f"{symbol}: 异步保存特征数据失败 - {str(e)}")
        return symbol, False

async def save_all_feature_data_async(data_dict: Dict[str, pd.DataFrame], max_workers: int = MAX_WORKERS) -> Dict[str, bool]:
    """
    异步保存所有特征数据
    
    Args:
        data_dict: {symbol: dataframe} 字典
        max_workers: 最大工作线程数
        
    Returns:
        {symbol: 是否成功} 字典
    """
    try:
        if not data_dict:
            logger.warning("没有数据需要保存")
            return {}
        
        logger.info(f"开始异步保存 {len(data_dict)} 个交易对的特征数据")
        
        # 创建线程池和异步任务
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [save_feature_data_async(symbol, df, executor) for symbol, df in data_dict.items()]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        save_results = {}
        successful_saves = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"保存任务异常: {str(result)}")
                continue
            
            symbol, success = result
            save_results[symbol] = success
            if success:
                successful_saves += 1
        
        logger.info(f"成功保存 {successful_saves}/{len(data_dict)} 个交易对的特征数据")
        return save_results
        
    except Exception as e:
        logger.error(f"异步保存所有特征数据失败: {str(e)}")
        return {}

def validate_data_consistency(labeled_data: Dict[str, pd.DataFrame], 
                            feature_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    验证标签数据和特征数据的一致性
    
    Args:
        labeled_data: 标签数据字典
        feature_data: 特征数据字典
        
    Returns:
        验证结果字典
    """
    try:
        validation_results = {}
        
        # 检查交易对一致性
        labeled_symbols = set(labeled_data.keys())
        feature_symbols = set(feature_data.keys())
        
        missing_in_feature = labeled_symbols - feature_symbols
        extra_in_feature = feature_symbols - labeled_symbols
        common_symbols = labeled_symbols & feature_symbols
        
        logger.info(f"数据一致性检查:")
        logger.info(f"标签数据交易对数: {len(labeled_symbols)}")
        logger.info(f"特征数据交易对数: {len(feature_symbols)}")
        logger.info(f"共同交易对数: {len(common_symbols)}")
        
        if missing_in_feature:
            logger.warning(f"特征数据中缺失的交易对: {missing_in_feature}")
        
        if extra_in_feature:
            logger.info(f"特征数据中额外的交易对: {extra_in_feature}")
        
        # 检查每个交易对的数据一致性
        for symbol in common_symbols:
            labeled_df = labeled_data[symbol]
            feature_df = feature_data[symbol]
            
            result = {
                'labeled_rows': len(labeled_df),
                'feature_rows': len(feature_df),
                'labeled_columns': len(labeled_df.columns),
                'feature_columns': len(feature_df.columns),
                'row_match': len(labeled_df) == len(feature_df),
                'column_increase': len(feature_df.columns) > len(labeled_df.columns),
                'new_columns': []
            }
            
            # 检查新增的列
            if set(labeled_df.columns).issubset(set(feature_df.columns)):
                new_columns = set(feature_df.columns) - set(labeled_df.columns)
                result['new_columns'] = list(new_columns)
            else:
                result['column_consistency'] = False
                logger.warning(f"{symbol}: 特征数据缺少标签数据中的某些列")
            
            validation_results[symbol] = result
        
        return validation_results
        
    except Exception as e:
        logger.error(f"验证数据一致性失败: {str(e)}")
        return {}





def cleanup_intermediate_files(temp_dir: Path = None):
    """
    清理中间临时文件
    
    Args:
        temp_dir: 临时文件目录
    """
    try:
        if temp_dir and temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                        logger.debug(f"删除临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败 {file}: {str(e)}")
        
        # 强制垃圾回收
        gc.collect()
        
    except Exception as e:
        logger.error(f"清理中间文件失败: {str(e)}")

def log_processing_summary(start_time: float, data_dict: Dict[str, pd.DataFrame], 
                         save_results: Dict[str, bool] = None):
    """
    记录处理汇总信息
    
    Args:
        start_time: 开始时间
        data_dict: 处理的数据字典
        save_results: 保存结果字典
    """
    try:
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("Step3 特征工程处理汇总:")
        logger.info("=" * 60)
        logger.info(f"处理时间: {processing_time:.2f} 秒")
        logger.info(f"处理交易对数: {len(data_dict)}")
        
        if data_dict:
            total_rows = sum(len(df) for df in data_dict.values())
            total_columns = sum(len(df.columns) for df in data_dict.values())
            avg_rows = total_rows / len(data_dict)
            avg_columns = total_columns / len(data_dict)
            
            logger.info(f"总数据行数: {total_rows:,}")
            logger.info(f"总列数: {total_columns}")
            logger.info(f"平均每个交易对行数: {avg_rows:.0f}")
            logger.info(f"平均每个交易对列数: {avg_columns:.1f}")
            
            if processing_time > 0:
                rows_per_second = total_rows / processing_time
                logger.info(f"处理速度: {rows_per_second:.0f} 行/秒")
        
        if save_results:
            successful_saves = sum(1 for success in save_results.values() if success)
            save_rate = successful_saves / len(save_results) * 100 if save_results else 0
            logger.info(f"保存成功率: {save_rate:.1f}% ({successful_saves}/{len(save_results)})")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"记录处理汇总失败: {str(e)}")

def get_memory_usage() -> str:
    """
    获取内存使用情况
    
    Returns:
        内存使用情况字符串
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A (psutil not available)"
    except Exception as e:
        return f"Error: {str(e)}"