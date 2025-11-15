"""
Step1 辅助函数模块
包含文件遍历、Parquet写入、分块处理等功能
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INPUT_CSV_PATH, RAW_DATA_PATH, FILE_SUFFIX, 
    PARQUET_ENGINE, PARQUET_COMPRESSION, MAX_WORKERS, BATCH_SIZE,
    ENABLE_1H_TO_1D_MERGE, MERGE_MAX_WORKERS, MERGE_BATCH_SIZE
)

logger = logging.getLogger(__name__)

def find_usdt_csv_files(input_dir: str = INPUT_CSV_PATH) -> List[Path]:
    """
    查找所有以-USDT结尾的CSV文件
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        CSV文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return []
    
    csv_files = []
    for file_path in input_path.glob("*.csv"):
        if file_path.stem.endswith(FILE_SUFFIX):
            csv_files.append(file_path)
    
    logger.info(f"找到 {len(csv_files)} 个USDT交易对CSV文件")
    return csv_files

def extract_symbol_from_filename(file_path: Path) -> str:
    """
    从文件名提取交易对符号
    
    Args:
        file_path: 文件路径
        
    Returns:
        交易对符号
    """
    # 假设文件名格式为: BTCUSDT-1h.csv 或 BTC-USDT.csv
    filename = file_path.stem
    
    # 移除可能的时间后缀
    if '-1h' in filename:
        filename = filename.replace('-1h', '')
    if '-4h' in filename:
        filename = filename.replace('-4h', '')
    if '-1d' in filename:
        filename = filename.replace('-1d', '')
    
    # 确保以USDT结尾
    if filename.endswith('-USDT'):
        return filename
    elif filename.endswith('USDT'):
        return filename
    else:
        return filename + 'USDT'

def save_to_parquet(df: pd.DataFrame, output_path: Path) -> bool:
    """
    保存DataFrame到Parquet文件
    
    Args:
        df: 要保存的DataFrame
        output_path: 输出文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为Parquet格式
        df.to_parquet(
            output_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False
        )
        
        logger.info(f"成功保存Parquet文件: {output_path}, 数据行数: {len(df)}")
        return True
        
    except Exception as e:
        logger.error(f"保存Parquet文件失败 {output_path}: {str(e)}")
        return False

def batch_save_parquet(data_list: List[Tuple[pd.DataFrame, Path]]) -> List[bool]:
    """
    批量保存多个DataFrame到Parquet文件
    
    Args:
        data_list: (DataFrame, 输出路径)元组列表
        
    Returns:
        保存结果列表
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_data = {
            executor.submit(save_to_parquet, df, path): (df, path)
            for df, path in data_list
        }
        
        for future in as_completed(future_to_data):
            df, path = future_to_data[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"批量保存出错 {path}: {str(e)}")
                results.append(False)
    
    return results

def process_dataframe_chunks(chunks: List[pd.DataFrame]) -> pd.DataFrame:
    """
    合并多个DataFrame块
    
    Args:
        chunks: DataFrame块列表
        
    Returns:
        合并后的DataFrame
    """
    if not chunks:
        return pd.DataFrame()
    
    try:
        # 合并所有块
        combined_df = pd.concat(chunks, ignore_index=True)
        
        # 按时间排序
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # 去重
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        
        logger.info(f"成功合并 {len(chunks)} 个数据块，总行数: {len(combined_df)}")
        return combined_df
        
    except Exception as e:
        logger.error(f"合并数据块失败: {str(e)}")
        return pd.DataFrame()

def validate_dataframe(df: pd.DataFrame, symbol: str) -> bool:
    """
    验证DataFrame数据质量
    
    Args:
        df: 要验证的DataFrame
        symbol: 交易对符号
        
    Returns:
        是否通过验证
    """
    if df.empty:
        logger.warning(f"{symbol}: DataFrame为空")
        return False
    
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"{symbol}: 缺少必要列: {missing_columns}")
        return False
    
    # 检查数值列是否有效
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if df[col].isna().all():
            logger.warning(f"{symbol}: 列 {col} 全为空值")
            return False
        
        if (df[col] <= 0).any() and col != 'volume':  # volume可以为0
            logger.warning(f"{symbol}: 列 {col} 存在非正值")
    
    # 检查OHLC逻辑
    invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                   (df['high'] < df['close']) | (df['low'] > df['open']) | \
                   (df['low'] > df['close'])
    
    if invalid_ohlc.any():
        logger.warning(f"{symbol}: 存在无效的OHLC数据")
        # 删除无效行而不是拒绝整个数据集
        df = df[~invalid_ohlc].reset_index(drop=True)
    
    logger.info(f"{symbol}: 数据验证通过，有效行数: {len(df)}")
    return True

async def process_files_in_batches(file_paths: List[Path], batch_size: int = BATCH_SIZE):
    """
    分批处理文件列表
    
    Args:
        file_paths: 文件路径列表
        batch_size: 批次大小
        
    Yields:
        文件路径批次
    """
    for i in range(0, len(file_paths), batch_size):
        yield file_paths[i:i + batch_size]

def get_output_path(symbol: str, output_dir: Path = RAW_DATA_PATH) -> Path:
    """
    获取输出文件路径
    
    Args:
        symbol: 交易对符号
        output_dir: 输出目录
        
    Returns:
        输出文件路径
    """
    return output_dir / f"{symbol}.parquet"

def cleanup_temp_files(temp_dir: Path):
    """
    清理临时文件
    
    Args:
        temp_dir: 临时文件目录
    """
    if temp_dir.exists():
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            logger.error(f"清理临时目录失败 {temp_dir}: {str(e)}")

def log_processing_stats(processed_files: int, total_files: int, failed_files: int):
    """
    记录处理统计信息
    
    Args:
        processed_files: 成功处理的文件数
        total_files: 总文件数
        failed_files: 失败的文件数
    """
    success_rate = (processed_files / total_files * 100) if total_files > 0 else 0
    
    logger.info("=" * 50)
    logger.info("数据处理完成统计:")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功处理: {processed_files}")
    logger.info(f"处理失败: {failed_files}")
    logger.info(f"成功率: {success_rate:.2f}%")
    logger.info("=" * 50)


def merge_1h_to_1d_data(df_1h: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    将1小时数据合并为1天数据
    
    Args:
        df_1h: 1小时K线数据DataFrame
        symbol: 交易对符号
        
    Returns:
        合并后的1天数据DataFrame
    """
    try:
        # 确保时间列存在且为datetime类型
        if 'timestamp' not in df_1h.columns:
            logger.error(f"{symbol}: 缺少timestamp列")
            return None
            
        # 转换时间列为datetime类型
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        
        # 设置时间索引
        df_1h = df_1h.set_index('timestamp')
        
        # 定义完整的聚合规则
        agg_dict = {
            'symbol': 'first',                          # 交易对名称：取第一个
            'open': 'first',                            # 开盘价：取当天第一个小时的开盘价
            'high': 'max',                              # 最高价：取当天所有小时中的最高价
            'low': 'min',                               # 最低价：取当天所有小时中的最低价
            'close': 'last',                            # 收盘价：取当天最后一个小时的收盘价
            'volume': 'sum',                            # 成交量：累加当天所有小时的成交量
            'quote_volume': 'sum',                      # 成交额：累加当天所有小时的成交额
        }
        
        # 添加可选字段的聚合规则（如果存在的话）
        optional_fields = {
            'trade_num': 'sum',                         # 交易笔数：累加
            'taker_buy_base_asset_volume': 'sum',       # 主动买入成交量：累加
            'taker_buy_quote_asset_volume': 'sum',      # 主动买入成交额：累加
            'spread': 'mean',                           # 价差：取平均值
            'avg_price_1m': 'mean',                     # 1分钟平均价：取平均值
            'avg_price_5m': 'mean',                     # 5分钟平均价：取平均值
            'fundingrate': 'sum',                       # 资金费率：累加
            'is_trading': 'last',                       # 是否交易：取最后一个状态
            'price_change': 'sum',                      # 价格变化：累加
            'is_price_anomaly': 'max',                  # 是否价格异常：取最大值（有异常就标记为异常）
        }
        
        # 检查哪些可选字段存在，并添加到聚合字典中
        for field, agg_method in optional_fields.items():
            if field in df_1h.columns:
                agg_dict[field] = agg_method
        
        # 按天重采样并聚合
        df_1d = df_1h.resample('D').agg(agg_dict)
        
        # 移除空数据行
        df_1d = df_1d.dropna(subset=['open', 'high', 'low', 'close'])  # 只要求OHLC不为空
        
        # 重置索引，将timestamp重新作为列
        df_1d = df_1d.reset_index()
        
        # 确保symbol列存在
        if 'symbol' not in df_1d.columns or df_1d['symbol'].isna().any():
            df_1d['symbol'] = symbol
        
        logger.info(f"{symbol}: 1小时数据({len(df_1h)})合并为1天数据({len(df_1d)})")
        
        return df_1d
        
    except Exception as e:
        logger.error(f"{symbol}: 数据合并失败 - {str(e)}")
        return None


def process_single_file_merge(file_path: Path) -> Tuple[bool, str]:
    """
    处理单个文件的1小时到1天数据合并，直接覆盖原文件
    
    Args:
        file_path: Parquet文件路径
        
    Returns:
        (是否成功, 错误信息)
    """
    import shutil
    
    try:
        symbol = file_path.stem  # 获取文件名作为symbol
        
        # 读取1小时数据
        df_1h = pd.read_parquet(file_path, engine=PARQUET_ENGINE)
        
        if df_1h is None or df_1h.empty:
            return False, f"{symbol}: 1小时数据为空"
        
        # 合并为1天数据
        df_1d = merge_1h_to_1d_data(df_1h, symbol)
        
        if df_1d is None or df_1d.empty:
            return False, f"{symbol}: 合并后数据为空"
        
        # 创建临时备份文件（防止写入失败）
        backup_path = file_path.with_suffix('.backup')
        shutil.copy2(file_path, backup_path)
        
        try:
            # 直接覆盖原文件
            success = save_to_parquet(df_1d, file_path)
            
            if success:
                # 删除备份文件
                backup_path.unlink()
                return True, f"{symbol}: 合并成功并覆盖原文件"
            else:
                # 恢复备份文件
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()
                return False, f"{symbol}: 保存1天数据失败"
                
        except Exception as save_error:
            # 恢复备份文件
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()
            raise save_error
            
    except Exception as e:
        return False, f"处理文件{file_path}失败: {str(e)}"


def batch_merge_1h_to_1d(file_paths: List[Path], max_workers: int = MERGE_MAX_WORKERS) -> Tuple[int, int]:
    """
    批量处理1小时到1天数据合并（多线程）
    
    Args:
        file_paths: 要处理的Parquet文件路径列表
        max_workers: 最大线程数
        
    Returns:
        (成功数量, 失败数量)
    """
    if not file_paths:
        logger.warning("没有文件需要合并")
        return 0, 0
    
    logger.info(f"开始批量合并{len(file_paths)}个文件，使用{max_workers}个线程")
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file_merge, file_path): file_path 
            for file_path in file_paths
        }
        
        # 收集结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    logger.debug(message)
                else:
                    failed_count += 1
                    logger.warning(message)
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"处理文件{file_path}时发生异常: {str(e)}")
    
    logger.info(f"批量合并完成: 成功{success_count}, 失败{failed_count}")
    return success_count, failed_count


async def merge_all_1h_to_1d_async(input_dir: Path = RAW_DATA_PATH) -> bool:
    """
    异步合并所有1小时数据为1天数据（直接覆盖原文件）
    
    Args:
        input_dir: 输入目录（包含1小时数据的Parquet文件）
        
    Returns:
        是否全部成功
    """
    try:
        # 查找所有Parquet文件（现在所有文件都可能需要合并）
        parquet_files = list(input_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning("没有找到需要合并的数据文件")
            return True
        
        # 过滤掉备份文件和临时文件
        valid_files = []
        for file_path in parquet_files:
            if not file_path.suffix == '.backup' and not file_path.name.startswith('.'):
                valid_files.append(file_path)
        
        if not valid_files:
            logger.warning("没有找到有效的数据文件")
            return True
        
        logger.info(f"找到{len(valid_files)}个数据文件需要合并为1天数据")
        
        # 分批处理
        total_success = 0
        total_failed = 0
        
        for i in range(0, len(valid_files), MERGE_BATCH_SIZE):
            batch_files = valid_files[i:i + MERGE_BATCH_SIZE]
            logger.info(f"处理第{i//MERGE_BATCH_SIZE + 1}批，共{len(batch_files)}个文件")
            
            # 在线程池中处理当前批次
            loop = asyncio.get_event_loop()
            success_count, failed_count = await loop.run_in_executor(
                None, batch_merge_1h_to_1d, batch_files, MERGE_MAX_WORKERS
            )
            
            total_success += success_count
            total_failed += failed_count
        
        # 记录最终统计
        logger.info("=" * 50)
        logger.info("1小时到1天数据合并完成统计:")
        logger.info(f"总文件数: {len(valid_files)}")
        logger.info(f"成功合并: {total_success}")
        logger.info(f"合并失败: {total_failed}")
        logger.info(f"成功率: {(total_success/len(valid_files)*100):.2f}%")
        logger.info("=" * 50)
        
        return total_failed == 0
        
    except Exception as e:
        logger.error(f"异步合并过程发生错误: {str(e)}")
        return False