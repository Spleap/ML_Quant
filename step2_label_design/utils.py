"""
Step2 辅助函数模块
包含Z-score、截面排序、数据加载和保存等工具函数
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, LABELED_DATA_PATH, PARQUET_ENGINE, 
    PARQUET_COMPRESSION, MAX_WORKERS, BATCH_SIZE
)

logger = logging.getLogger(__name__)

def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    加载Step1生成的原始数据
    
    Returns:
        {symbol: dataframe} 字典
    """
    try:
        raw_data = {}
        parquet_files = list(RAW_DATA_PATH.glob("*.parquet"))
        
        if not parquet_files:
            logger.error(f"在 {RAW_DATA_PATH} 中未找到任何Parquet文件")
            return {}
        
        logger.info(f"找到 {len(parquet_files)} 个原始数据文件")
        
        for file_path in parquet_files:
            try:
                symbol = file_path.stem  # 文件名即为交易对符号
                df = pd.read_parquet(file_path, engine=PARQUET_ENGINE)
                
                if not df.empty:
                    raw_data[symbol] = df
                    logger.info(f"加载 {symbol}: {len(df)} 行数据")
                else:
                    logger.warning(f"{symbol}: 数据为空")
                    
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {str(e)}")
        
        logger.info(f"成功加载 {len(raw_data)} 个交易对的原始数据")
        return raw_data
        
    except Exception as e:
        logger.error(f"加载原始数据失败: {str(e)}")
        return {}

async def load_raw_data_async() -> Dict[str, pd.DataFrame]:
    """
    异步加载Step1生成的原始数据
    
    Returns:
        {symbol: dataframe} 字典
    """
    try:
        parquet_files = list(RAW_DATA_PATH.glob("*.parquet"))
        
        if not parquet_files:
            logger.error(f"在 {RAW_DATA_PATH} 中未找到任何Parquet文件")
            return {}
        
        logger.info(f"开始异步加载 {len(parquet_files)} 个原始数据文件")
        
        async def load_single_file(file_path: Path) -> Tuple[str, Optional[pd.DataFrame]]:
            """加载单个文件"""
            try:
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None, 
                    pd.read_parquet, 
                    file_path, 
                    PARQUET_ENGINE
                )
                
                symbol = file_path.stem
                if not df.empty:
                    logger.info(f"异步加载 {symbol}: {len(df)} 行数据")
                    return symbol, df
                else:
                    logger.warning(f"{symbol}: 数据为空")
                    return symbol, None
                    
            except Exception as e:
                logger.error(f"异步加载文件失败 {file_path}: {str(e)}")
                return file_path.stem, None
        
        # 创建异步任务
        tasks = [load_single_file(file_path) for file_path in parquet_files]
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        raw_data = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"异步加载异常: {str(result)}")
                continue
            
            symbol, df = result
            if df is not None:
                raw_data[symbol] = df
        
        logger.info(f"异步加载完成，成功加载 {len(raw_data)} 个交易对")
        return raw_data
        
    except Exception as e:
        logger.error(f"异步加载原始数据失败: {str(e)}")
        return {}

def save_labeled_data(labeled_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """
    保存标签数据到labeled_data目录
    
    Args:
        labeled_data: {symbol: labeled_dataframe} 字典
        
    Returns:
        {symbol: success} 保存结果字典
    """
    try:
        # 确保输出目录存在
        LABELED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        save_results = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {}
            
            for symbol, df in labeled_data.items():
                output_path = LABELED_DATA_PATH / f"{symbol}.parquet"
                future = executor.submit(save_single_labeled_file, df, output_path, symbol)
                future_to_symbol[future] = symbol
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    save_results[symbol] = success
                except Exception as e:
                    logger.error(f"{symbol}: 保存异常 - {str(e)}")
                    save_results[symbol] = False
        
        success_count = sum(save_results.values())
        logger.info(f"标签数据保存完成: {success_count}/{len(labeled_data)} 成功")
        
        return save_results
        
    except Exception as e:
        logger.error(f"保存标签数据失败: {str(e)}")
        return {symbol: False for symbol in labeled_data.keys()}

def save_single_labeled_file(df: pd.DataFrame, output_path: Path, symbol: str) -> bool:
    """
    保存单个标签数据文件
    
    Args:
        df: 标签数据DataFrame
        output_path: 输出路径
        symbol: 交易对符号
        
    Returns:
        是否保存成功
    """
    try:
        # 清理因标签计算不足导致的缺失值
        # 标签列（label）的缺失值需要清理，其他列的缺失值保留
        cleaned_df = df.copy()
        
        if 'label' in cleaned_df.columns:
            # 删除标签列为NaN的行（通常是末尾8行，因为8期收益率无法计算）
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=['label'])
            removed_rows = initial_rows - len(cleaned_df)
            
            if removed_rows > 0:
                logger.info(f"{symbol}: 清理了 {removed_rows} 行因标签计算不足导致的缺失数据")
        
        # 确保清理后仍有数据
        if cleaned_df.empty:
            logger.warning(f"{symbol}: 清理后数据为空，跳过保存")
            return False
        
        cleaned_df.to_parquet(
            output_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False
        )
        
        logger.info(f"{symbol}: 标签数据保存成功 -> {output_path} ({len(cleaned_df)} 行)")
        return True
        
    except Exception as e:
        logger.error(f"{symbol}: 标签数据保存失败 -> {output_path}: {str(e)}")
        return False

async def save_labeled_data_async(labeled_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
    """
    异步保存标签数据
    
    Args:
        labeled_data: {symbol: labeled_dataframe} 字典
        
    Returns:
        {symbol: success} 保存结果字典
    """
    try:
        # 确保输出目录存在
        LABELED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        async def save_single_async(symbol: str, df: pd.DataFrame) -> Tuple[str, bool]:
            """异步保存单个文件"""
            try:
                output_path = LABELED_DATA_PATH / f"{symbol}.parquet"
                loop = asyncio.get_event_loop()
                
                success = await loop.run_in_executor(
                    None,
                    save_single_labeled_file,
                    df,
                    output_path,
                    symbol
                )
                
                return symbol, success
                
            except Exception as e:
                logger.error(f"{symbol}: 异步保存失败 - {str(e)}")
                return symbol, False
        
        # 创建异步任务
        tasks = [save_single_async(symbol, df) for symbol, df in labeled_data.items()]
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        save_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"异步保存异常: {str(result)}")
                continue
            
            symbol, success = result
            save_results[symbol] = success
        
        success_count = sum(save_results.values())
        logger.info(f"异步保存完成: {success_count}/{len(labeled_data)} 成功")
        
        return save_results
        
    except Exception as e:
        logger.error(f"异步保存标签数据失败: {str(e)}")
        return {symbol: False for symbol in labeled_data.keys()}

def validate_data_consistency(raw_data: Dict[str, pd.DataFrame], labeled_data: Dict[str, pd.DataFrame]) -> bool:
    """
    验证原始数据和标签数据的一致性
    
    Args:
        raw_data: 原始数据字典
        labeled_data: 标签数据字典
        
    Returns:
        是否一致
    """
    try:
        # 检查交易对数量
        if len(raw_data) != len(labeled_data):
            logger.error(f"交易对数量不一致: 原始数据{len(raw_data)}, 标签数据{len(labeled_data)}")
            return False
        
        # 检查每个交易对
        inconsistent_symbols = []
        
        for symbol in raw_data.keys():
            if symbol not in labeled_data:
                logger.error(f"标签数据中缺少交易对: {symbol}")
                inconsistent_symbols.append(symbol)
                continue
            
            raw_df = raw_data[symbol]
            labeled_df = labeled_data[symbol]
            
            # 检查行数（标签数据可能因为计算未来收益率而少几行）
            if len(labeled_df) > len(raw_df):
                logger.error(f"{symbol}: 标签数据行数({len(labeled_df)})大于原始数据({len(raw_df)})")
                inconsistent_symbols.append(symbol)
                continue
            
            # 检查必要列是否存在
            if 'label' not in labeled_df.columns:
                logger.error(f"{symbol}: 标签数据缺少label列")
                inconsistent_symbols.append(symbol)
                continue
            
            # 检查原始列是否保留
            original_columns = set(raw_df.columns)
            labeled_columns = set(labeled_df.columns) - {'label'}  # 排除新增的label列
            
            missing_columns = original_columns - labeled_columns
            if missing_columns:
                logger.warning(f"{symbol}: 标签数据缺少原始列: {missing_columns}")
        
        if inconsistent_symbols:
            logger.error(f"数据一致性检查失败，不一致的交易对: {inconsistent_symbols}")
            return False
        
        logger.info("数据一致性检查通过")
        return True
        
    except Exception as e:
        logger.error(f"数据一致性检查失败: {str(e)}")
        return False





def cleanup_intermediate_files():
    """清理中间文件"""
    try:
        # 这里可以添加清理逻辑，比如删除临时文件等
        logger.info("中间文件清理完成")
    except Exception as e:
        logger.error(f"清理中间文件失败: {str(e)}")

def log_processing_summary(total_symbols: int, successful_symbols: int, failed_symbols: int):
    """
    记录处理汇总信息
    
    Args:
        total_symbols: 总交易对数
        successful_symbols: 成功处理数
        failed_symbols: 失败处理数
    """
    success_rate = (successful_symbols / total_symbols * 100) if total_symbols > 0 else 0
    
    logger.info("=" * 50)
    logger.info("Step2 标签设计完成统计:")
    logger.info(f"总交易对数: {total_symbols}")
    logger.info(f"成功处理: {successful_symbols}")
    logger.info(f"处理失败: {failed_symbols}")
    logger.info(f"成功率: {success_rate:.2f}%")
    logger.info("=" * 50)