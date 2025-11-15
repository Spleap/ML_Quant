"""
Step1 主执行文件
原始数据处理：CSV -> Parquet
每次运行前清空raw_data目录
"""
import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_PATH, clear_directory, ensure_data_dirs,
    LOG_LEVEL, LOG_FORMAT, MAX_WORKERS,
    ENABLE_1H_TO_1D_MERGE
)
from step1_data_processing.file_reader import CSVFileReader
from step1_data_processing.data_cleaner import DataCleaner
from step1_data_processing.utils import (
    find_usdt_csv_files, extract_symbol_from_filename, save_to_parquet,
    process_dataframe_chunks, validate_dataframe, get_output_path,
    log_processing_stats, process_files_in_batches, merge_all_1h_to_1d_async
)

# 配置日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

async def process_single_file(csv_reader: CSVFileReader, file_path: Path) -> bool:
    """
    处理单个CSV文件
    
    Args:
        csv_reader: CSV读取器实例
        file_path: CSV文件路径
        
    Returns:
        是否处理成功
    """
    try:
        # 提取交易对符号
        symbol = extract_symbol_from_filename(file_path)
        logger.info(f"开始处理 {symbol}: {file_path}")
        
        # 读取CSV文件
        df = await csv_reader.read_csv_async(file_path)
        
        if df is None or df.empty:
            logger.warning(f"{symbol}: 读取失败或数据为空")
            return False
        
        # 数据清洗
        cleaner = DataCleaner()
        df = cleaner.clean_dataframe(df, symbol)
        
        if df is None or df.empty:
            logger.warning(f"{symbol}: 数据清洗后为空")
            return False
        
        # 验证数据质量
        if not validate_dataframe(df, symbol):
            logger.warning(f"{symbol}: 数据验证失败")
            return False
        
        # 获取输出路径
        output_path = get_output_path(symbol)
        
        # 保存为Parquet
        success = save_to_parquet(df, output_path)
        
        if success:
            logger.info(f"{symbol}: 处理完成，输出到 {output_path}")
            return True
        else:
            logger.error(f"{symbol}: 保存失败")
            return False
            
    except Exception as e:
        logger.error(f"处理文件失败 {file_path}: {str(e)}")
        return False

async def process_large_file(csv_reader: CSVFileReader, file_path: Path) -> bool:
    """
    处理大型CSV文件（分块读取）
    
    Args:
        csv_reader: CSV读取器实例
        file_path: CSV文件路径
        
    Returns:
        是否处理成功
    """
    try:
        symbol = extract_symbol_from_filename(file_path)
        logger.info(f"开始分块处理大文件 {symbol}: {file_path}")
        
        # 分块读取
        chunks = csv_reader.read_csv_chunked(file_path)
        
        if not chunks:
            logger.warning(f"{symbol}: 分块读取失败")
            return False
        
        # 合并数据块
        df = process_dataframe_chunks(chunks)
        
        if df.empty:
            logger.warning(f"{symbol}: 合并后数据为空")
            return False
        
        # 数据清洗
        cleaner = DataCleaner()
        df = cleaner.clean_dataframe(df, symbol)
        
        if df is None or df.empty:
            logger.warning(f"{symbol}: 数据清洗后为空")
            return False
        
        # 验证数据质量
        if not validate_dataframe(df, symbol):
            logger.warning(f"{symbol}: 数据验证失败")
            return False
        
        # 获取输出路径
        output_path = get_output_path(symbol)
        
        # 保存为Parquet
        success = save_to_parquet(df, output_path)
        
        if success:
            logger.info(f"{symbol}: 大文件处理完成，输出到 {output_path}")
            return True
        else:
            logger.error(f"{symbol}: 保存失败")
            return False
            
    except Exception as e:
        logger.error(f"处理大文件失败 {file_path}: {str(e)}")
        return False

async def main():
    """主执行函数"""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("开始执行 Step1: 原始数据处理")
    logger.info("=" * 60)
    
    try:
        # 确保数据目录存在
        ensure_data_dirs()
        
        # 清空raw_data目录
        logger.info("清空raw_data目录...")
        clear_directory(RAW_DATA_PATH)
        
        # 查找所有USDT CSV文件
        csv_files = find_usdt_csv_files()
        
        if not csv_files:
            logger.error("未找到任何USDT交易对CSV文件")
            return
        
        logger.info(f"找到 {len(csv_files)} 个CSV文件待处理")
        
        # 创建CSV读取器
        csv_reader = CSVFileReader(max_workers=MAX_WORKERS)
        
        # 统计变量
        processed_count = 0
        failed_count = 0
        
        # 分批处理文件
        batch_num = 1
        async for file_batch in process_files_in_batches(csv_files):
            logger.info(f"处理第 {batch_num} 批文件，共 {len(file_batch)} 个文件")
            
            # 并发处理当前批次的文件
            tasks = []
            for file_path in file_batch:
                # 检查文件大小，决定处理方式
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb > 100:  # 大于100MB的文件使用分块处理
                    task = process_large_file(csv_reader, file_path)
                else:
                    task = process_single_file(csv_reader, file_path)
                
                tasks.append(task)
            
            # 等待当前批次完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"处理异常: {file_batch[i]} - {str(result)}")
                    failed_count += 1
                elif result:
                    processed_count += 1
                else:
                    failed_count += 1
            
            logger.info(f"第 {batch_num} 批处理完成")
            batch_num += 1
        
        # 记录处理统计
        log_processing_stats(processed_count, len(csv_files), failed_count)
        
        # 验证输出
        output_files = list(RAW_DATA_PATH.glob("*.parquet"))
        logger.info(f"生成的Parquet文件数量: {len(output_files)}")
        
        if len(output_files) != processed_count:
            logger.warning("输出文件数量与处理成功数量不匹配")
        
        # 1小时到1天数据合并（如果启用）
        if ENABLE_1H_TO_1D_MERGE:
            logger.info("=" * 60)
            logger.info("开始执行 1小时到1天数据合并")
            logger.info("=" * 60)
            
            merge_start_time = time.time()
            
            try:
                # 执行异步合并
                merge_success = await merge_all_1h_to_1d_async(RAW_DATA_PATH)
                
                merge_end_time = time.time()
                merge_duration = merge_end_time - merge_start_time
                
                if merge_success:
                    logger.info(f"1小时到1天数据合并完成，耗时: {merge_duration:.2f} 秒")
                    
                    # 统计合并后的文件
                    merged_files = list(RAW_DATA_PATH.glob("*_1d.parquet"))
                    logger.info(f"生成的1天数据文件数量: {len(merged_files)}")
                else:
                    logger.warning(f"1小时到1天数据合并部分失败，耗时: {merge_duration:.2f} 秒")
                    
            except Exception as e:
                logger.error(f"1小时到1天数据合并过程发生错误: {str(e)}")
        else:
            logger.info("1小时到1天数据合并功能已禁用")
        
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"Step1 总耗时: {total_duration:.2f} 秒")
        
    except Exception as e:
        logger.error(f"Step1 执行失败: {str(e)}")
        raise
    
    finally:
        logger.info("Step1 执行结束")

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())