"""
数据清洗模块
参考temp.py中的数据清洗逻辑，对原始CSV数据进行清洗和标准化
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗器，负责对原始CSV数据进行清洗和标准化"""
    
    def __init__(self):
        self.required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        # 列名映射字典
        self.column_mapping = {
            'candle_begin_time': 'timestamp',
            'Candle_begin_time': 'timestamp',
            'time': 'timestamp',
            'datetime': 'timestamp'
        }
        
    def clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        清洗单个DataFrame的数据
        
        参数:
            df: 原始DataFrame
            symbol: 交易对符号
            
        返回:
            清洗后的DataFrame
        """
        try:
            logger.info(f"开始清洗数据: {symbol}")
            
            # 0. 列名映射和标准化
            df = self._standardize_columns(df)
            
            # 1. 基础数据验证
            df = self._validate_basic_data(df, symbol)
            
            # 2. 删除重复的时间点记录
            df = self._remove_duplicates(df)
            
            # 3. 构建连续时间序列并填补缺失时间点
            df = self._fill_missing_timestamps(df)
            
            # 4. 填充缺失的价格数据
            df = self._fill_missing_prices(df)
            
            # 5. 填充缺失的成交量数据
            df = self._fill_missing_volumes(df)
            
            # 6. 添加交易活动标记
            df = self._add_trading_flags(df)
            
            # 7. 数据类型优化
            df = self._optimize_data_types(df, symbol)
            
            logger.info(f"数据清洗完成: {symbol}, 最终数据量: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"数据清洗失败: {symbol}, 错误: {str(e)}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名，将不同的时间列名映射为timestamp
        
        参数:
            df: 原始DataFrame
            
        返回:
            列名标准化后的DataFrame
        """
        # 重命名列
        df = df.rename(columns=self.column_mapping)
        
        # 确保列名为小写
        df.columns = df.columns.str.lower()
        
        return df
    
    def _validate_basic_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """基础数据验证"""
        if df.empty:
            raise ValueError(f"数据为空: {symbol}")
        
        # 检查必需列
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复的时间点记录，保留最后一条"""
        original_len = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        if len(df) < original_len:
            logger.warning(f"删除了 {original_len - len(df)} 条重复记录")
        
        return df
    
    def _fill_missing_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建连续时间序列，填补缺失的时间点"""
        if len(df) == 0:
            return df
        
        # 获取时间范围
        first_time = df['timestamp'].min()
        last_time = df['timestamp'].max()
        
        # 构建1小时连续时间序列
        time_range = pd.DataFrame({
            'timestamp': pd.date_range(start=first_time, end=last_time, freq='1h')
        })
        
        # 合并数据，确保时间连续性
        df = pd.merge(
            left=time_range, 
            right=df, 
            on='timestamp', 
            how='left', 
            sort=True
        )
        
        # 再次去重并排序
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _fill_missing_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失的价格数据"""
        # 首先填充收盘价（向前填充）
        df['close'] = df['close'].ffill()
        
        # 用收盘价填充其他价格字段的缺失值
        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['low'] = df['low'].fillna(df['close'])
        
        # 确保high >= max(open, close), low <= min(open, close)
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def _fill_missing_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失的成交量数据"""
        volume_columns = ['volume', 'quote_volume', 'trade_num', 
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in volume_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 确保成交量为非负数
        for col in volume_columns:
            if col in df.columns:
                df[col] = np.maximum(df[col], 0)
        
        return df
    
    def _add_trading_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交易活动标记"""
        # 标记是否有交易活动（成交量大于0）
        df['is_trading'] = np.where(df['volume'] > 0, 1, 0).astype(np.int8)
        
        # 计算价格变化率
        df['price_change'] = df['close'].pct_change()
        
        # 标记异常价格变化（变化率超过50%）
        df['is_price_anomaly'] = np.where(
            np.abs(df['price_change']) > 0.5, 1, 0
        ).astype(np.int8)
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        # 符号列使用分类类型
        df['symbol'] = pd.Categorical([symbol] * len(df))
        
        # 价格列转换为float32（如果精度允许）
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
        # 成交量列转换为适当的整数类型
        volume_columns = ['volume', 'quote_volume', 'trade_num', 
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in volume_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float32')
        
        return df
    
    def get_cleaning_stats(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """获取清洗统计信息"""
        return {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'missing_timestamps_filled': len(cleaned_df) - len(original_df),
            'price_anomalies': cleaned_df['is_price_anomaly'].sum() if 'is_price_anomaly' in cleaned_df.columns else 0,
            'trading_periods': cleaned_df['is_trading'].sum() if 'is_trading' in cleaned_df.columns else 0,
            'non_trading_periods': len(cleaned_df) - cleaned_df['is_trading'].sum() if 'is_trading' in cleaned_df.columns else 0
        }

def clean_single_file(file_path: str, symbol: str) -> pd.DataFrame:
    """
    清洗单个CSV文件的便捷函数
    
    参数:
        file_path: CSV文件路径
        symbol: 交易对符号
        
    返回:
        清洗后的DataFrame
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(
            file_path, 
            encoding='gbk', 
            parse_dates=['candle_begin_time'], 
            skiprows=1
        )
        
        # 创建清洗器并清洗数据
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_dataframe(df, symbol)
        
        # 记录清洗统计
        stats = cleaner.get_cleaning_stats(df, cleaned_df)
        logger.info(f"清洗统计 {symbol}: {stats}")
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"文件清洗失败: {file_path}, 错误: {str(e)}")
        raise