"""
RSI (Relative Strength Index) 相对强弱指数因子
计算指定周期的RSI值
"""
import pandas as pd
import numpy as np

def RSI(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算RSI相对强弱指数
    
    Args:
        df: 包含OHLCV数据的DataFrame，必须包含'close'列
        period: RSI计算周期，默认14
        
    Returns:
        RSI值的Series
    """
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame必须包含'close'列")
        
        if len(df) < period + 1:
            # 数据不足，返回NaN
            return pd.Series([np.nan] * len(df), index=df.index)
        
        close = df['close'].copy()
        
        # 计算价格变化
        delta = close.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均收益和平均损失（使用指数移动平均）
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # 计算相对强度
        rs = avg_gain / avg_loss
        
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        
        # 处理除零情况
        rsi = rsi.fillna(50)  # 当avg_loss为0时，RSI设为50
        
        return rsi
        
    except Exception as e:
        # 发生错误时返回NaN序列
        return pd.Series([np.nan] * len(df), index=df.index)

def calculate(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI计算的别名函数，与RSI函数功能相同
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: RSI计算周期
        
    Returns:
        RSI值的Series
    """
    return RSI(df, period)