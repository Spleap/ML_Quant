"""
Step1 高性能CSV文件读取模块
支持GBK编码、跳过第一行、分块读取和多线程处理
"""
import pandas as pd
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from typing import List, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CSV_ENCODING, SKIP_ROWS, CHUNK_SIZE, MAX_WORKERS

logger = logging.getLogger(__name__)

class CSVFileReader:
    """高性能CSV文件读取器"""
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def read_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        读取单个CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame或None（如果读取失败）
        """
        try:
            logger.info(f"开始读取文件: {file_path}")
            
            # 使用pandas读取CSV，支持GBK编码
            df = pd.read_csv(
                file_path,
                encoding=CSV_ENCODING,
                skiprows=SKIP_ROWS,
                low_memory=False
            )
            
            # 基本数据清洗 - 只删除关键列的空值
            # 定义关键列（OHLCV数据）
            key_columns = ['open', 'high', 'low', 'close', 'volume']
            existing_key_columns = [col for col in key_columns if col in df.columns]
            
            if existing_key_columns:
                # 只删除关键列有空值的行
                df = df.dropna(subset=existing_key_columns)
            else:
                # 如果没有关键列，删除所有列都为空的行
                df = df.dropna(how='all')
            
            # 确保时间列存在并转换格式
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
            
            # 确保数值列为正确类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"成功读取文件: {file_path}, 数据行数: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None
    
    def read_csv_chunked(self, file_path: Path, chunk_size: int = CHUNK_SIZE) -> List[pd.DataFrame]:
        """
        分块读取大型CSV文件
        
        Args:
            file_path: CSV文件路径
            chunk_size: 每块大小
            
        Returns:
            DataFrame列表
        """
        try:
            logger.info(f"开始分块读取大文件: {file_path}")
            chunks = []
            
            for chunk in pd.read_csv(
                file_path,
                encoding=CSV_ENCODING,
                skiprows=SKIP_ROWS,
                chunksize=chunk_size,
                low_memory=False
            ):
                # 基本清洗 - 只删除关键列的空值
                key_columns = ['open', 'high', 'low', 'close', 'volume']
                existing_key_columns = [col for col in key_columns if col in chunk.columns]
                
                if existing_key_columns:
                    chunk = chunk.dropna(subset=existing_key_columns)
                else:
                    chunk = chunk.dropna(how='all')
                
                # 时间列处理
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                elif 'time' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['time'])
                    chunk = chunk.drop('time', axis=1)
                
                # 数值列处理
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in chunk.columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                
                chunks.append(chunk)
            
            logger.info(f"分块读取完成: {file_path}, 总块数: {len(chunks)}")
            return chunks
            
        except Exception as e:
            logger.error(f"分块读取失败 {file_path}: {str(e)}")
            return []
    
    async def read_csv_async(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        异步读取CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            DataFrame或None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.read_csv_file, file_path)
    
    async def read_multiple_csv_async(self, file_paths: List[Path]) -> List[tuple]:
        """
        异步批量读取多个CSV文件
        
        Args:
            file_paths: CSV文件路径列表
            
        Returns:
            (文件路径, DataFrame)元组列表
        """
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def read_with_semaphore(file_path):
            async with semaphore:
                df = await self.read_csv_async(file_path)
                return (file_path, df)
        
        tasks = [read_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for result in results:
            if not isinstance(result, Exception) and result[1] is not None:
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"异步读取出错: {str(result)}")
        
        return valid_results
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)